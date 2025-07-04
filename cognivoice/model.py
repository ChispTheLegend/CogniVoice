import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import WhisperForAudioClassification, BertModel


disvoice_feature_names = ['static-Articulation', 'static-Phonation',
       'static-RepLearning', 'static-Prosody', 'static-Phonological',
       'dynamic-Articulation', 'dynamic-Phonation', 'dynamic-RepLearning',
       'dynamic-Prosody', 'dynamic-Phonological']

class GradMulConst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)

class Whisper(nn.Module):
    def __init__(self, args, num_labels=2):
        super().__init__()

        num_labels = 2 if args.task == 'cls' else 1
        self.args = args
        self.audio_model = WhisperForAudioClassification.from_pretrained(args.method)

        final_output_dim = self.audio_model.config.classifier_proj_size
        self.config = self.audio_model.config

        if self.args.use_disvoice:
            self.disvoice_projector = nn.Linear(len(disvoice_feature_names), len(disvoice_feature_names))
            final_output_dim += len(disvoice_feature_names)

        if self.args.use_metadata:
            self.metadata_projector = nn.Linear(3, 3)
            final_output_dim += 3

        if self.args.use_text:
            self.en_text_model = BertModel.from_pretrained('bert-base-uncased')
            self.en_text_mlp = nn.Linear(self.en_text_model.config.hidden_size, self.en_text_model.config.hidden_size)
            self.cn_text_model = BertModel.from_pretrained('bert-base-chinese')
            self.cn_text_mlp = nn.Linear(self.cn_text_model.config.hidden_size, self.cn_text_model.config.hidden_size)
            classifier_dropout = (
                self.en_text_model.config.classifier_dropout if self.en_text_model.config.classifier_dropout is not None else self.en_text_model.config.hidden_dropout_prob
            )
            self.text_dropout = nn.Dropout(classifier_dropout)

            # Initialize weights and apply final processing
            self.en_text_model.post_init()
            self.cn_text_model.post_init()
            final_output_dim += self.en_text_model.config.hidden_size

        if self.args.use_llama2:
            self.llama2_model = BertModel.from_pretrained('bert-base-uncased')
            classifier_dropout = (
                self.llama2_model.config.classifier_dropout if self.llama2_model.config.classifier_dropout is not None else self.llama2_model.config.hidden_dropout_prob
            )
            self.llama2_dropout = nn.Dropout(classifier_dropout)

            # Initialize weights and apply final processing
            self.llama2_model.post_init()
            final_output_dim += self.llama2_model.config.hidden_size

        self.dropout = nn.Dropout(self.args.dropout)

        self.final_projector = nn.Sequential(
            nn.Linear(final_output_dim, final_output_dim//2),
            nn.ReLU(),
            nn.Linear(final_output_dim//2, final_output_dim//4),
        )
        self.classifier = nn.Linear(final_output_dim//4, num_labels)

    def forward(self, input_features=None, head_mask=None, encoder_outputs=None, labels=None, 
        output_attentions=None, output_hidden_states=None, return_dict=None, disvoice_features=None, metadata=None, lang=None,
        text_input_ids=None, text_attention_mask=None, llama2_input_ids=None, llama2_attention_mask=None):

        # Audio model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.audio_model.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = torch.stack(encoder_outputs, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.audio_model.projector(hidden_states)
        final_output = hidden_states.mean(dim=1)    
        final_output = self.dropout(final_output)     # Pooled output

        # Disvoice feature
        if self.args.use_disvoice:
            disvoice_output = self.disvoice_projector(disvoice_features)
            final_output = torch.cat([final_output, disvoice_output], dim=1)

        if self.args.use_metadata:
            metadata_output = self.metadata_projector(metadata)
            final_output = torch.cat([final_output, metadata_output], dim=1)

        if self.args.use_text:
            text_output = torch.zeros(text_input_ids.shape[0], self.en_text_model.config.hidden_size, device=text_input_ids.device)

            ## English
            en_id = torch.where(lang == 0)
            en_text_out = self.en_text_model(
                text_input_ids[en_id],
                attention_mask=text_attention_mask[en_id],
            )
            en_text_out = en_text_out[1]
            en_text_out = self.en_text_mlp(en_text_out)
            en_text_out = self.text_dropout(en_text_out)
            text_output[en_id] = en_text_out

            ## Chinese
            cn_id = torch.where(lang == 1)
            cn_text_out = self.cn_text_model(
                text_input_ids[cn_id],
                attention_mask=text_attention_mask[cn_id],
            )
            cn_text_out = cn_text_out[1]
            cn_text_out = self.cn_text_mlp(cn_text_out)
            cn_text_out = self.text_dropout(cn_text_out)
            text_output[cn_id] = cn_text_out
            
            final_output = torch.cat([final_output, text_output], dim=1)

        if self.args.use_llama2:
            llama2_output = self.llama2_model(
                llama2_input_ids,
                attention_mask=llama2_attention_mask,
            )
            llama2_output = llama2_output[1]
            llama2_output = self.llama2_dropout(llama2_output)
            final_output = torch.cat([final_output, llama2_output], dim=1)

        final_output = self.final_projector(final_output)
        logits = self.classifier(final_output)

        loss = None
        if labels is not None:
            if self.args.task == 'cls':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            else:
                loss_fct = nn.MSELoss()
                # if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class WhisperPoe(nn.Module):
    def __init__(self, args, num_labels=2):
        super().__init__()

        self.args = args
        self.audio_model = WhisperForAudioClassification.from_pretrained(args.method)

        final_output_dim = self.audio_model.config.classifier_proj_size
        self.config = self.audio_model.config
        self.audio_identity = nn.Identity()
        self.audio_only_classifier = nn.Sequential(
            nn.Linear(self.audio_model.config.classifier_proj_size, self.audio_model.config.classifier_proj_size//2),
            nn.ReLU(),
            nn.Linear(self.audio_model.config.classifier_proj_size//2, num_labels)
        )

        if self.args.use_disvoice:
            self.disvoice_projector = nn.Linear(len(disvoice_feature_names), len(disvoice_feature_names))
            final_output_dim += len(disvoice_feature_names)

            self.disvoice_identity = nn.Identity()
            self.disvoice_only_classifier = nn.Sequential(
                nn.Linear(len(disvoice_feature_names), len(disvoice_feature_names)//2),
                nn.ReLU(),
                nn.Linear(len(disvoice_feature_names)//2, num_labels)
            )

        if self.args.use_metadata:
            self.metadata_projector = nn.Linear(3, 3)
            final_output_dim += 3

        if self.args.use_text:
            self.en_text_model = BertModel.from_pretrained('bert-base-uncased')
            self.en_text_mlp = nn.Linear(self.en_text_model.config.hidden_size, self.en_text_model.config.hidden_size)
            self.cn_text_model = BertModel.from_pretrained('bert-base-chinese')
            self.cn_text_mlp = nn.Linear(self.cn_text_model.config.hidden_size, self.cn_text_model.config.hidden_size)
            classifier_dropout = (
                self.en_text_model.config.classifier_dropout if self.en_text_model.config.classifier_dropout is not None else self.en_text_model.config.hidden_dropout_prob
            )
            self.text_dropout = nn.Dropout(classifier_dropout)

            # Initialize weights and apply final processing
            self.en_text_model.post_init()
            self.cn_text_model.post_init()
            final_output_dim += self.en_text_model.config.hidden_size

            self.text_identity = nn.Identity()
            self.text_only_classifier = nn.Sequential(
                nn.Linear(self.en_text_model.config.hidden_size, self.en_text_model.config.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.en_text_model.config.hidden_size//2, num_labels)
            )

        if self.args.use_llama2:
            self.llama2_model = BertModel.from_pretrained('bert-base-uncased')
            classifier_dropout = (
                self.llama2_model.config.classifier_dropout if self.llama2_model.config.classifier_dropout is not None else self.llama2_model.config.hidden_dropout_prob
            )
            self.llama2_dropout = nn.Dropout(classifier_dropout)

            # Initialize weights and apply final processing
            self.llama2_model.post_init()
            final_output_dim += self.llama2_model.config.hidden_size

            self.llama2_identity = nn.Identity()
            self.llama2_only_classifier = nn.Sequential(
                nn.Linear(self.llama2_model.config.hidden_size, self.llama2_model.config.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.llama2_model.config.hidden_size//2, num_labels)
            )

        self.final_projector = nn.Sequential(
            nn.Linear(final_output_dim, final_output_dim//2),
            nn.ReLU(),
            nn.Linear(final_output_dim//2, final_output_dim//4),
        )
        self.classifier = nn.Linear(final_output_dim//4, num_labels)

    def forward(self, input_features=None, head_mask=None, encoder_outputs=None, labels=None, 
        output_attentions=None, output_hidden_states=None, return_dict=None, disvoice_features=None, metadata=None, lang=None,
        text_input_ids=None, text_attention_mask=None, llama2_input_ids=None, llama2_attention_mask=None):

        # Audio model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.audio_model.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = torch.stack(encoder_outputs, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        branch_logits = []

        hidden_states = self.audio_model.projector(hidden_states)
        final_output = hidden_states.mean(dim=1)    # Pooled output
        if self.args.use_poe:
            grad_mul_const(self.audio_identity(final_output), 0.0)
            audio_only_logits = self.audio_only_classifier(final_output)
            branch_logits.append(audio_only_logits)

        # Disvoice feature
        if self.args.use_disvoice:
            disvoice_output = self.disvoice_projector(disvoice_features)
            final_output = torch.cat([final_output, disvoice_output], dim=1)
            if self.args.use_poe:
                grad_mul_const(self.disvoice_identity(disvoice_output), 0.0)
                disvoice_only_logits = self.disvoice_only_classifier(disvoice_output)
                branch_logits.append(disvoice_only_logits)

        if self.args.use_metadata:
            metadata_output = self.metadata_projector(metadata)
            final_output = torch.cat([final_output, metadata_output], dim=1)
            if self.args.use_poe:
                grad_mul_const(self.identity(metadata_output), 0.0)
                metadata_only_logits = self.metadata_only_classifier(metadata_output)
                branch_logits.append(metadata_only_logits)

        if self.args.use_text:
            text_output = torch.zeros(text_input_ids.shape[0], self.en_text_model.config.hidden_size, device=text_input_ids.device)

            ## English
            en_id = torch.where(lang == 0)
            en_text_out = self.en_text_model(
                text_input_ids[en_id],
                attention_mask=text_attention_mask[en_id],
            )
            en_text_out = en_text_out[1]
            en_text_out = self.en_text_mlp(en_text_out)
            en_text_out = self.text_dropout(en_text_out)
            text_output[en_id] = en_text_out

            ## Chinese
            cn_id = torch.where(lang == 1)
            cn_text_out = self.cn_text_model(
                text_input_ids[cn_id],
                attention_mask=text_attention_mask[cn_id],
            )
            cn_text_out = cn_text_out[1]
            cn_text_out = self.cn_text_mlp(cn_text_out)
            cn_text_out = self.text_dropout(cn_text_out)
            text_output[cn_id] = cn_text_out
            
            final_output = torch.cat([final_output, text_output], dim=1)
            if self.args.use_poe:
                grad_mul_const(self.text_identity(text_output), 0.0)
                text_only_logits = self.text_only_classifier(text_output)
                branch_logits.append(text_only_logits)

        if self.args.use_llama2:
            llama2_output = self.llama2_model(
                llama2_input_ids,
                attention_mask=llama2_attention_mask,
            )
            llama2_output = llama2_output[1]
            llama2_output = self.llama2_dropout(llama2_output)
            final_output = torch.cat([final_output, llama2_output], dim=1)
            if self.args.use_poe:
                grad_mul_const(self.llama2_identity(llama2_output), 0.0)
                llama2_only_logits = self.llama2_only_classifier(llama2_output)
                branch_logits.append(llama2_only_logits)

        final_output = self.final_projector(final_output)
        logits = self.classifier(final_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.args.use_poe:
                if self.training:
                    # POE loss
                    branch_logits = [self.args.poe_alpha * F.log_softmax(i, dim=1) for i in branch_logits]
                    logits_combined = F.log_softmax(logits, dim=1) + torch.stack(branch_logits).sum(0)
                    loss_ori = loss_fct(logits_combined, labels.view(-1))

                    # Bias only loss of each branch
                    branch_loss = []
                    for logi in branch_logits:
                        branch_loss.append(loss_fct(logi, labels))

                    # Total loss
                    loss = loss_ori + self.args.poe_alpha * torch.tensor(branch_loss).sum()

                else:
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        #6.27.25 forward method return dict with raw logits from each expert
        individual_expert_raw_logits = {}

        # 1. Multi-feature Expert's Logits (from the concatenated features)
        # 'logits' already holds this from its previous calculation
        individual_expert_raw_logits['multi_feature'] = logits

        # 2. Uni-feature Experts' Logits
        # Collect logits for each expert *if* they are enabled
        # We need to ensure the necessary features (e.g., disvoice_features)
        # and the expert classifiers (e.g., self.disvoice_only_classifier)
        # were instantiated and their forward passes executed.

        # Audio Only Expert
        # The 'final_output' variable, at the point after audio processing
        # but before concatenation with other modalities, represents the audio feature.
        # We'll re-calculate its specific classifier output here to ensure we get its raw logits.
        
               
        # ---- START DEBUG PRINTS ----
        print(f"DEBUG: Shape of hidden_states: {hidden_states.shape}")
        
        if hasattr(self.audio_model, 'projector') and isinstance(self.audio_model.projector, torch.nn.Linear):
            print(f"DEBUG: self.audio_model.projector.in_features: {self.audio_model.projector.in_features}")
        else:
            print(f"DEBUG: self.audio_model.projector is not a standard Linear layer or not initialized.")
            
        print(f"DEBUG: self.audio_model.config.d_model: {self.audio_model.config.d_model}")
        # ---- END DEBUG PRINTS ----       
        audio_features_for_expert = self.audio_model.projector(hidden_states).mean(dim=1)
        individual_expert_raw_logits['audio_only'] = self.audio_only_classifier(audio_features_for_expert)

        # Disvoice Only Expert
        if self.args.use_disvoice:
            # Ensure disvoice_output is available (it is if use_disvoice is true)
            # Re-run its specific classifier to get raw logits
            individual_expert_raw_logits['disvoice_only'] = self.disvoice_only_classifier(self.disvoice_projector(disvoice_features))

        # Metadata Only Expert
        if self.args.use_metadata:
            # IMPORTANT: Confirm self.metadata_only_classifier exists in __init__
            # If not, you'll need to add it or remove this block.
            # Assuming it exists and metadata_output is available:
            individual_expert_raw_logits['metadata_only'] = self.metadata_only_classifier(self.metadata_projector(metadata))


        # Text Only Expert
        if self.args.use_text:
            # 'text_output' should contain the pooled BERT features.
            # Re-run its specific classifier to get raw logits.
            # If text_output is only populated for specific lang IDs, it might need adjustment
            # but generally, the `text_output` variable should contain the relevant features.
            individual_expert_raw_logits['text_only'] = self.text_only_classifier(text_output)


        # Llama2 Only Expert
        if self.args.use_llama2:
            # 'llama2_output' should contain the pooled Llama2 features.
            # Re-run its specific classifier to get raw logits.
            individual_expert_raw_logits['llama2_only'] = self.llama2_only_classifier(llama2_output)


        # Determine the primary 'logits' to return for the standard Trainer's use.
        # When PoE is active, the combined log-probabilities are the most representative.
        if self.args.use_poe:
            # Combine individual expert log-probabilities (scaled by alpha)
            # and add to the multi-feature expert's log-probabilities.
            # This is the fusion logic, applied here for the final output.
            
            # Make sure to filter out None values for experts that might not be enabled
            active_branch_log_probs_eval = []
            for expert_name, expert_raw_logi in individual_expert_raw_logits.items():
                if expert_name != 'multi_feature' and expert_raw_logi is not None:
                    active_branch_log_probs_eval.append(self.args.poe_alpha * F.log_softmax(expert_raw_logi, dim=1))

            main_logits_for_trainer = F.log_softmax(individual_expert_raw_logits['multi_feature'], dim=1)
            
            if active_branch_log_probs_eval: # Check if there are any uni-feature experts contributing
                logits_to_return_as_primary = main_logits_for_trainer + torch.stack(active_branch_log_probs_eval).sum(0)
            else: # If no uni-feature experts are active, just use multi-feature
                logits_to_return_as_primary = main_logits_for_trainer
                
            # If regression task, convert back from log-prob
            if self.args.task == 'reg':
                logits_to_return_as_primary = torch.exp(logits_to_return_as_primary) # Convert log-probs back to scale for MSE
                # Note: This is an assumption. If your regression uses raw PoE scores, not probabilities, adjust here.
                # If your regression loss is applied directly to sum of log-scores, this exp() might not be needed.
                # For classification, LogSoftmax is what CrossEntropyLoss expects internally.

        else: # If not using PoE, the primary logits are just from the multi-feature expert
            logits_to_return_as_primary = logits


        # Return a dictionary. The Hugging Face Trainer can handle dictionaries
        # returned from the model's forward pass.
        return {
            "loss": loss, # The loss calculated (PoE loss if use_poe, or standard loss)
            "logits": logits_to_primary, # This will be the main prediction for compute_metrics
            "individual_expert_logits": individual_expert_raw_logits, # All expert raw logits for debugging
            # Include other outputs if needed by your Trainer setup or for further debugging
            "hidden_states": encoder_outputs.hidden_states if return_dict else None,
            "attentions": encoder_outputs.attentions if return_dict else None,
        }
        #END CHANGES        
               
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
