import math
import torch
from torch import nn
import torch.nn.functional as F
from .SubNets.FeatureNets import BERTEncoder
from .SubNets.AlignNets import AlignSubNet

class CDPR(nn.Module):
    def __init__(self, args):
        super(CDPR, self).__init__()
        
        self.text_feats_dim = args.text_feats_dim
        self.video_feats_dim = args.video_feats_dim
        self.audio_feats_dim = args.audio_feats_dim
        self.hidden_feats_dim = args.weight_hidden_dim
        self.weight_dropout = args.weight_dropout
        self.temperature = args.temperature
        self.need_aligned = args.need_aligned
        self.aligned_method = args.aligned_method
        self.num_modalities = args.num_modalities
        self.num_labels = args.num_labels

        self.text_subnet = BERTEncoder(args)
        self.proj_t = nn.Sequential(nn.Linear(self.text_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))
        self.proj_v = nn.Sequential(nn.Linear(self.video_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))
        self.proj_a = nn.Sequential(nn.Linear(self.audio_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))

        self.privatized_t = nn.Sequential(nn.Linear(self.hidden_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))        
        self.privatized_v = nn.Sequential(nn.Linear(self.hidden_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))
        self.privatized_a = nn.Sequential(nn.Linear(self.hidden_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))
        self.shared = nn.Sequential(nn.Linear(self.hidden_feats_dim, self.hidden_feats_dim), nn.ReLU(), nn.Dropout(self.weight_dropout))

        if self.need_aligned:
            self.alignNet = AlignSubNet(args, self.aligned_method)

        self.raw_fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_feats_dim * 3, self.hidden_feats_dim),
            nn.Dropout(self.weight_dropout)
        )
        self.consensus_scale = nn.Parameter(torch.zeros(1))
        self.intuition_norm = nn.LayerNorm(self.hidden_feats_dim)

        self.inconsistency_fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_feats_dim * 3, self.hidden_feats_dim),
            nn.ReLU(),
            nn.Dropout(self.weight_dropout)
        )
        self.sem_norm_layer = nn.LayerNorm(self.hidden_feats_dim)

        self.consensus_extractor = nn.Sequential(
            nn.Linear(self.hidden_feats_dim * 3, self.hidden_feats_dim),
            nn.ReLU(),
            nn.Dropout(self.weight_dropout)
        )

        self.semantic_compressor = nn.Sequential(
            nn.Linear(self.hidden_feats_dim * 3, self.hidden_feats_dim),
            nn.ReLU(),
            nn.Dropout(self.weight_dropout)
        )
   
        self.uncertainty_projector = nn.Sequential(
            nn.Linear(3, self.hidden_feats_dim),
            nn.ReLU(),
            nn.Dropout(self.weight_dropout)
        )

        self.conflict_prototype = nn.Parameter(torch.randn(1, self.hidden_feats_dim))
        self.stat_bias_layer = nn.Linear(4, 1)
        self.js_gate_linear = nn.Linear(1, self.hidden_feats_dim)
        self.gate_generator = nn.Linear(self.hidden_feats_dim, 1)
        self.trust_generator = nn.Linear(self.hidden_feats_dim, 3)
        self.reasoning_classifier = nn.Linear(self.hidden_feats_dim, self.num_labels)

        self.text_classifier = nn.Linear(self.hidden_feats_dim, self.num_labels)
        self.video_classifier = nn.Linear(self.hidden_feats_dim, self.num_labels)
        self.audio_classifier = nn.Linear(self.hidden_feats_dim, self.num_labels)
        self.final_classifier = nn.Linear(self.hidden_feats_dim, self.num_labels)

        self.fusion_bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, text_feats, video_feats, audio_feats):

        seq_t = self.text_subnet(text_feats)
        seq_v = video_feats.float()
        seq_a = audio_feats.float()     
        if self.need_aligned:
            vec_t, vec_a, vec_v = self.alignNet(seq_t, seq_a, seq_v)
        h_t = self.proj_t(vec_t)
        h_v = self.proj_v(vec_v)
        h_a = self.proj_a(vec_a)

        # Private-shared components
        shared_t = self.shared(h_t)
        shared_v = self.shared(h_v)
        shared_a = self.shared(h_a) 
        private_t = self.privatized_t(h_t)
        private_v = self.privatized_v(h_v)
        private_a = self.privatized_a(h_a)

        # Consistency Fusion
        raw_feats = torch.cat([h_t, h_v, h_a], dim=-1)
        z_raw = self.raw_fusion_layer(raw_feats)
        syn_tv = shared_t * shared_v
        syn_ta = shared_t * shared_a
        syn_va = shared_v * shared_a
        sim_feat = torch.cat([syn_tv, syn_ta, syn_va], dim=-1)
        e_consensus = self.consensus_extractor(sim_feat)
        z_intuition = z_raw + self.consensus_scale * e_consensus
        z_intuition = self.intuition_norm(z_intuition)

        # CCM: Inconsistency Fusion
        c = (private_t + private_v + private_a) / 3.0
        h_diff_tc = torch.abs(private_t - c)
        h_diff_vc = torch.abs(private_v - c)
        h_diff_ac = torch.abs(private_a - c)
        diff_feat = torch.cat([h_diff_tc, h_diff_vc, h_diff_ac], dim=-1)
        e_sem = self.inconsistency_fusion_layer(diff_feat) # refined difference vector
        e_sem_norm = F.normalize(e_sem, p=2, dim=-1)
        proto_norm = F.normalize(self.conflict_prototype, p=2, dim=-1)
        energy_semantic = torch.matmul(e_sem_norm, proto_norm.t()) * self.temperature

        logits_t = self.text_classifier(private_t)[:, 0, :]
        logits_v = self.video_classifier(private_v).mean(dim=1, keepdim=False)
        logits_a = self.audio_classifier(private_a).mean(dim=1, keepdim=False)
        logits_list = [logits_t, logits_v, logits_a]
        probs = [F.softmax(l, dim=-1) for l in logits_list]

        # conflict level: High JS
        avg_prob = torch.stack(probs, dim=1).mean(dim=1)     
        js_loss = 0.0
        for p in probs:
            kl = F.kl_div((avg_prob + 1e-8).log(), p, reduction='none').sum(dim=-1)  # quantify the inconsistency between each modality and the average result
            js_loss += kl
        js_val = js_loss / len(probs)
        js_val = js_val.unsqueeze(-1).detach()
        js_val = js_val / (math.log(self.num_modalities) + 1e-8)

        # uncertainty: Low Entropy
        max_entropy  = math.log(self.num_labels) + 1e-8
        ent_t = -torch.sum(probs[0] * torch.log(probs[0] + 1e-8), dim=-1, keepdim=True) / max_entropy
        ent_v = -torch.sum(probs[1] * torch.log(probs[1] + 1e-8), dim=-1, keepdim=True) / max_entropy
        ent_a = -torch.sum(probs[2] * torch.log(probs[2] + 1e-8), dim=-1, keepdim=True) / max_entropy
        uncertainty_feat = torch.cat([ent_t, ent_v, ent_a], dim=-1)
        s_stat = torch.cat([js_val, uncertainty_feat], dim=-1)
        stat_bias = self.stat_bias_layer(s_stat).unsqueeze(1)

        energy = energy_semantic + stat_bias
        attn_score = torch.sigmoid(energy)
        v_reflect = attn_score * e_sem # obtain the conflict vector

        scaling_uncertainty = self.uncertainty_projector(uncertainty_feat)
        trust_input = v_reflect + scaling_uncertainty.unsqueeze(1)
        trust_weights = F.softmax(self.trust_generator(trust_input), dim=-1)
        w_t = trust_weights[:, :, 0].unsqueeze(-1)
        w_v = trust_weights[:, :, 1].unsqueeze(-1)
        w_a = trust_weights[:, :, 2].unsqueeze(-1)
        z_reasoning = w_t * private_t + w_v * private_v + w_a * private_a 
        reasoning_logits = self.reasoning_classifier(z_reasoning.mean(dim=1))

        sem_energy = torch.norm(v_reflect, p=2, dim=-1, keepdim=True)
        gate_sem = torch.tanh(sem_energy)
        gate_stat = js_val.unsqueeze(1)
        gate_logits = gate_sem + gate_stat + self.fusion_bias
        lambda_gate = torch.sigmoid(gate_logits)

        z_final = (1 - lambda_gate) * z_intuition + lambda_gate * z_reasoning
        z_final = z_final.mean(dim=1, keepdim=False)
        final_logits = self.final_classifier(z_final)

        outputs = {
            "logits": final_logits,
            "unimodal_logits": logits_list,
            "reasoning_logits": reasoning_logits,
            "shared_t": shared_t, "shared_v": shared_v, "shared_a": shared_a,
            "private_t": private_t, "private_v": private_v, "private_a": private_a
            }
        
        return outputs