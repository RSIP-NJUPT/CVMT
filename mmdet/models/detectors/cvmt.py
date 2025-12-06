from typing import Optional, Dict, List, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn.init import normal_
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from mmdet.structures import DetDataSample
ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
@MODELS.register_module()
class CVMT(DeformableDETR):
    """Implementation of the CVMT: Complex-valued mix transformer for SAR ship detection.
    Adapted from the official GitHub repository <https://github.com/RSIP-NJUPT/CVMT>.

    Args:
        num_encoder_queries (int): Number of queries used by the encoder stage.
        num_fused_queries (int): Number of queries fused from original features.
        dn_cfg (ConfigDict or dict, optional): Configuration for denoising query generator.
            Default is None, which uses default denoising settings.
    """

    def __init__(
        self,
        num_encoder_queries,
        num_fused_queries,
        *args,
        dn_cfg: OptConfigType = None,
        **kwargs
    ) -> None:
        # store custom query counts before calling parent init
        self.num_encoder_queries = num_encoder_queries
        self.num_fused_queries = num_fused_queries
        super().__init__(*args, **kwargs)
        # DINO requires two-stage and box refinement enabled
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        # Prepare denoising query generator if configuration is provided
        if dn_cfg is not None:
            # ensure user does not override key args
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                ('The three keyword args num_classes, embed_dims, and '
                 'num_matching_queries are set in detector.__init__(), '
                 'users should not set them in dn_cfg config.')
            # fill in required entries
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            # total queries = encoder + fused ones
            dn_cfg['num_matching_queries'] = (
                self.num_encoder_queries + self.num_fused_queries
            )
        # instantiate the denoising query generator module
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers excluding backbone, neck, and bbox_head."""
        # sine-based positional encoding for multi-scale features
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        # Transformer encoder and decoder modules
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)

        # embedding dimension used across modules
        self.embed_dims = self.encoder.embed_dims
        # learnable query embeddings for both encoder and fused queries
        self.query_embedding = nn.Embedding(
            self.num_encoder_queries + self.num_fused_queries,
            self.embed_dims
        )
        # helper layers for position transformation
        self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)

        # verify embed_dims equals 2 * num_feats
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f'embed_dims should be exactly 2 times of num_feats. '
            f'Found {self.embed_dims} and {num_feats}.'
        )

        # level embedding for multi-scale features
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        # linear + norm to transform encoder memory if needed
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and related components."""
        # call parent class weight init
        super(DeformableDETR, self).init_weights()
        # Xavier initialize encoder and decoder parameters
        for coder in (self.encoder, self.decoder):
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        # init deformable attention layers
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        # init custom linear layers and embeddings
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Full forward pass through encoder, query prep and decoder."""
        # prepare encoder & partial decoder inputs
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        # flatten original multi-scale image features
        feat_flatten, num_point_list, fused_feat_flatten = self.process_img_feats(img_feats)
        encoder_outputs_dict['fused_feat_flatten'] = fused_feat_flatten
        encoder_outputs_dict['num_point_list'] = num_point_list
        # prepare decoder inputs (queries, reference points, etc.)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict,
            batch_data_samples=batch_data_samples
        )
        decoder_inputs_dict.update(tmp_dec_in)
        # run decoder
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        fused_feat_flatten: Tensor,  # original flattened features
        num_point_list: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """Generate queries, reference points, and other inputs for decoder."""
        bs, _, c = memory.shape
        # classification feature size from last decoder layer
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features
        # generate proposals from encoder memory and original(fused) features
        output_enc_memory, output_enc_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        output_fused, output_fused_proposals = self.gen_encoder_output_proposals(
            fused_feat_flatten, memory_mask, spatial_shapes
        )
        # compute class logits for proposals
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ](output_enc_memory)
        fused_output_class = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ](output_fused)

        # compute bbox regression and add proposals shift
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](
                output_enc_memory
            ) + output_enc_proposals
        )
        fused_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](
                output_fused
            ) + output_fused_proposals
        )

        # select top-k proposals by classification score
        enc_topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0],
            k=self.num_encoder_queries,
            dim=1
        )[1]
        # gather corresponding scores and coords
        enc_topk_score = torch.gather(
            enc_outputs_class, 1,
            enc_topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)
        )
        enc_topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            enc_topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        # apply sigmoid to get normalized box centers/sizes
        enc_topk_coords = enc_topk_coords_unact.sigmoid()
        # detach unactivated coords for regression targets
        enc_topk_coords_unact = enc_topk_coords_unact.detach()

        # repeat same for original-feature proposals
        fused_topk_indices = torch.topk(
            fused_output_class.max(-1)[0],
            k=self.num_fused_queries,
            dim=1
        )[1]
        fused_topk_score = torch.gather(
            fused_output_class, 1,
            fused_topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)
        )
        fused_topk_coords_unact = torch.gather(
            fused_outputs_coord_unact, 1,
            fused_topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        fused_topk_coords = fused_topk_coords_unact.sigmoid()
        fused_topk_coords_unact = fused_topk_coords_unact.detach()

        # concatenate encoder and original proposals
        topk_coords_unact = torch.cat([enc_topk_coords_unact, fused_topk_coords_unact], dim=1)
        topk_coords = torch.cat([enc_topk_coords, fused_topk_coords], dim=1)
        topk_score = torch.cat([enc_topk_score, fused_topk_score], dim=1)

        # prepare query embeddings (content queries)
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.training:
            # generate denoising queries for training
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = (
                self.dn_query_generator(batch_data_samples)
            )
            # prepend denoising queries to content queries
            query = torch.cat([dn_label_query, query], dim=1)
            # combine dn bboxes and topk proposals as reference
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        # normalize reference points to [0,1]
        reference_points = reference_points.sigmoid()

        # assemble decoder inputs and head inputs
        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask
        )
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta
        ) if self.training else {}
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Dict:
        """Run the Transformer decoder to refine queries and predict boxes."""
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs
        )

        # dummy operation to ensure label embedding is used (no-op addition)
        if len(query) == self.num_queries:
            inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states,
            references=list(references)
        )
        return decoder_outputs_dict

    def process_img_feats(
        self,
        img_feats: Tensor
    ):
        """Flatten multi-scale feature maps into sequence for Transformer input.

        Args:
            img_feats (list[Tensor]): feature maps with shapes (B, C, H, W)

        Returns:
            feat_flatten: list of flattened tensors (B, H*W, C)
            num_point_list: list of H*W sizes per level
            ori_feat_flatten: concatenated features (B, \sum H*W, C)
        """
        feat_flatten = []
        num_point_list = []
        for feat in img_feats:
            batch_size, c, h, w = feat.shape
            flattened_feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(flattened_feat)
            num_point_list.append(h * w)
        fused_feat_flatten = torch.cat(feat_flatten, dim=1)
        return feat_flatten, num_point_list, fused_feat_flatten

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.
        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.

        """
        if mode == 'loss':
            input1 = inputs[0].repeat(1,3,1,1)
            input2 = inputs[1].repeat(1,3,1,1)
            inputs = [input1, input2]
            loss = self.loss(inputs, data_samples)
            return loss

        elif mode == 'predict':
            input1 = inputs[0].repeat(1,3,1,1)
            input2 = inputs[1].repeat(1,3,1,1)
            inputs = [input1, input2]
            predit = self.predict(inputs, data_samples)
            return predit

        elif mode == 'tensor':
            input1 = inputs[0].repeat(1,3,1,1)
            input2 = inputs[1].repeat(1,3,1,1)
            inputs = [input1, input2]
            return self._forward(inputs, data_samples)

        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
