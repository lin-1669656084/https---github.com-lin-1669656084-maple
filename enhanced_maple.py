# enhanced_maple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnhancedMaPLe(nn.Module):
    def __init__(self, input_dim=28 * 28, num_classes=10, prompt_dim=512,
                 mask_ratio=0.3, num_layers=3, dropout_rate=0.3,
                 use_attention=True, use_residual=True):
        super(EnhancedMaPLe, self).__init__()
        self.prompt_dim = prompt_dim
        self.mask_ratio = mask_ratio
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Multi-scale feature extraction
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )

        # Calculate conv output size
        conv_output_size = 128 * 4 * 4  # 2048

        # Initial embedding with multiple pathways
        self.embedding_layers = nn.ModuleList([
            nn.Linear(input_dim, prompt_dim // 2),  # Raw pixel pathway
            nn.Linear(conv_output_size, prompt_dim // 2)  # Conv feature pathway
        ])

        # Learnable prompt tokens (similar to vision transformers)
        self.prompt_tokens = nn.Parameter(torch.randn(1, 16, prompt_dim))
        nn.init.normal_(self.prompt_tokens, std=0.02)

        # Multi-head attention for prompt interaction
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=prompt_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(prompt_dim)

        # Multi-layer prompt processing with residual connections
        self.prompt_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(prompt_dim * 2, prompt_dim),
                nn.Dropout(dropout_rate)
            )
            self.prompt_layers.append(layer)

        # Layer normalization for each prompt layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(prompt_dim) for _ in range(num_layers)
        ])

        # Adaptive masking with multiple mask types
        self.mask_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim // 4),
                nn.ReLU(),
                nn.Linear(prompt_dim // 4, prompt_dim),
                nn.Sigmoid()
            ) for _ in range(3)  # 3 different mask types
        ])

        # Feature fusion and classification
        self.feature_fusion = nn.Sequential(
            nn.Linear(prompt_dim * 3, prompt_dim),  # Fuse 3 masked features
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prompt_dim, prompt_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Multi-head classifier for ensemble-like prediction
        self.classifiers = nn.ModuleList([
            nn.Linear(prompt_dim // 2, num_classes) for _ in range(3)
        ])

        # Final ensemble weight
        self.ensemble_weight = nn.Parameter(torch.ones(3))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, training=True):
        B = x.size(0)

        # Extract features through different pathways
        # Pathway 1: Raw pixels
        x_raw = x.view(B, -1)
        raw_features = self.embedding_layers[0](x_raw)

        # Pathway 2: Convolutional features
        conv_features = self.conv_features(x)
        conv_features = conv_features.view(B, -1)
        conv_features = self.embedding_layers[1](conv_features)

        # Combine pathways
        x_combined = torch.cat([raw_features, conv_features], dim=1)  # [B, prompt_dim]

        # Add learnable prompt tokens
        prompt_tokens = self.prompt_tokens.expand(B, -1, -1)  # [B, 16, prompt_dim]
        x_with_prompts = torch.cat([
            x_combined.unsqueeze(1),  # [B, 1, prompt_dim]
            prompt_tokens
        ], dim=1)  # [B, 17, prompt_dim]

        # Multi-head attention for prompt interaction
        if self.use_attention:
            attended, _ = self.attention(x_with_prompts, x_with_prompts, x_with_prompts)
            x_combined = self.attention_norm(attended[:, 0, :] + x_combined)  # Use first token + residual

        # Multi-layer prompt processing with residual connections
        current_features = x_combined
        for i, (layer, norm) in enumerate(zip(self.prompt_layers, self.layer_norms)):
            residual = current_features
            processed = layer(current_features)
            if self.use_residual:
                current_features = norm(processed + residual)
            else:
                current_features = norm(processed)

        # Generate multiple adaptive masks
        masked_features = []
        for mask_gen in self.mask_generators:
            mask = mask_gen(current_features)
            # Add noise during training for regularization
            if training and self.training:
                noise = torch.randn_like(mask) * 0.1
                mask = torch.clamp(mask + noise, 0, 1)
            masked_feature = current_features * mask
            masked_features.append(masked_feature)

        # Fuse all masked features
        fused_features = torch.cat(masked_features, dim=1)
        final_features = self.feature_fusion(fused_features)

        # Multi-head classification with ensemble
        logits_list = []
        for classifier in self.classifiers:
            logits = classifier(final_features)
            logits_list.append(logits)

        # Ensemble prediction
        ensemble_weights = F.softmax(self.ensemble_weight, dim=0)
        final_logits = sum(w * logits for w, logits in zip(ensemble_weights, logits_list))

        # Return individual logits during training for auxiliary loss
        if training and self.training:
            return final_logits, logits_list
        else:
            return final_logits

    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        if not self.use_attention:
            return None

        B = x.size(0)
        x_raw = x.view(B, -1)
        raw_features = self.embedding_layers[0](x_raw)
        conv_features = self.conv_features(x).view(B, -1)
        conv_features = self.embedding_layers[1](conv_features)
        x_combined = torch.cat([raw_features, conv_features], dim=1)

        prompt_tokens = self.prompt_tokens.expand(B, -1, -1)
        x_with_prompts = torch.cat([x_combined.unsqueeze(1), prompt_tokens], dim=1)

        _, attention_weights = self.attention(x_with_prompts, x_with_prompts, x_with_prompts)
        return attention_weights


# Enhanced training loss with auxiliary objectives
class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(EnhancedLoss, self).__init__()
        self.alpha = alpha  # Main loss weight
        self.beta = beta  # Auxiliary loss weight
        self.gamma = gamma  # Diversity loss weight
        self.main_criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_logits, aux_logits_list = outputs

            # Main loss
            main_loss = self.main_criterion(main_logits, targets)

            # Auxiliary losses (encourage diversity among classifiers)
            aux_loss = 0
            for aux_logits in aux_logits_list:
                aux_loss += self.main_criterion(aux_logits, targets)
            aux_loss /= len(aux_logits_list)

            # Diversity loss (encourage different classifiers to be different)
            diversity_loss = 0
            for i in range(len(aux_logits_list)):
                for j in range(i + 1, len(aux_logits_list)):
                    # KL divergence between different classifier outputs
                    p_i = F.softmax(aux_logits_list[i], dim=1)
                    p_j = F.softmax(aux_logits_list[j], dim=1)
                    diversity_loss += F.kl_div(p_i.log(), p_j, reduction='batchmean')

            total_loss = (self.alpha * main_loss +
                          self.beta * aux_loss -
                          self.gamma * diversity_loss)

            return total_loss
        else:
            return self.main_criterion(outputs, targets)


# Updated Config class
class EnhancedConfig:
    def __init__(self):
        self.device = "cuda"
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 20
        self.prompt_dim = 512
        self.mask_ratio = 0.3
        self.num_layers = 3
        self.dropout_rate = 0.3
        self.use_attention = True
        self.use_residual = True
        self.weight_decay = 1e-4
        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.5