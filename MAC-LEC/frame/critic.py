
import torch
from torch import Tensor, nn

from models.encoders import LSTMEncoder, TransformerEncoder, MLPEncoder, TemporalEncoder
from models.decoders import LSTMDecoder, MLPDecoder, BiLSTM_Attention, GRUDecoder,OuterProductDecoder


class Actor(nn.Module):
    """
    Design of Actor Part in Reinforcement Learning Actor-Critic Algorithm.

    Include ``Encoder`` and ``Decoder``. The ``Encoder`` is used to map the
    observed data to the embedding space S={s1, · · · , sd}.
    The ``Decoder`` maps the state space S^(S_hat) to the action space A.

    Parameters
    ----------
    input_dim: int
        dimension of input data, number of variables, number of DAG node.
    embed_dim: int, default: 256
        dimension of embedding space S.
    encoder_blocks: int, default: 3
        Effective when `encoder`='transformer'.
        Design for the neural network structure of the Transformer encoder,
        each block is composed of a multi-head attention network and
        feed-forward neural networks.
    encoder_heads: int, default: 8
        Effective when `encoder_name`='transformer'.
        head number of multi-head attention network,
    encoder_name: str, default: 'transformer'
        Indicates type of encoder, one of [`transformer`, `lstm`, `mlp`]
    decoder_name: str, default: 'lstm'
        Indicates type of decoder, one of [`lstm`, `mlp`]
    """

    ENCODER_HIDDEN_DIM = 1024

    def __init__(self, input_dim, embed_dim=256, max_length=5,
                 encoder_name='transformer',
                 encoder_blocks=3,
                 encoder_heads=8,
                 decoder_name='bilstm',
                 device=None) -> None:
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder_blocks = encoder_blocks
        self.encoder_heads = encoder_heads
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.max_length = max_length
        self.device = device
        self.float()
        self._instantiation()

    def _instantiation(self):
        if self.encoder_name.lower() == 'transformer':
            self.encoder = TransformerEncoder(input_dim=self.input_dim,
                                              embed_dim=self.embed_dim,
                                              hidden_dim=self.ENCODER_HIDDEN_DIM,
                                              heads=self.encoder_heads,
                                              blocks=self.encoder_blocks,
                                              max_length = self.max_length,
                                              device=self.device).float()
        elif self.encoder_name.lower() == 'lstm':
            self.encoder = LSTMEncoder(input_dim=self.input_dim,
                                embed_dim=self.embed_dim,
                                device=self.device).float()
        elif self.encoder_name.lower() == 'mlp':
            self.encoder = MLPEncoder(input_dim=self.input_dim,
                              embed_dim=self.embed_dim,
                              hidden_dim=self.ENCODER_HIDDEN_DIM,
                              device=self.device).float()
        elif self.encoder_name.lower() == 'temporal':
            self.encoder = TemporalEncoder(hidden_dim=self.embed_dim,
                                           num_nodes=5,
                                           time_steps=self.input_dim).to(self.device).float()

        else:
            raise ValueError(f'Invalid encoder type, expected one of '
                             f'[`transformer`, `lstm`, `mlp`], but got'
                             f'``{self.encoder_name}``.')

        if self.decoder_name.lower() == 'lstm':
            self.decoder = LSTMDecoder(input_dim=self.embed_dim,
                                hidden_dim=self.embed_dim,
                                device=self.device).float()
        elif self.decoder_name.lower() == 'mlp':
            self.decoder = MLPDecoder(input_dim=self.embed_dim,
                              hidden_dim=self.embed_dim,
                              device=self.device).float()
        elif self.decoder_name.lower() == 'bilstm':
            self.decoder = BiLSTM_Attention(input_dim=self.embed_dim,
                              hidden_dim=self.embed_dim,
                              device=self.device).float()
        elif self.decoder_name.lower() == 'gru':
            self.decoder = GRUDecoder(hidden_dim=self.embed_dim,
                                      num_regions=5).to(self.device).float()
        elif self.decoder_name.lower() == 'outer':
            self.decoder = OuterProductDecoder(hidden_dim=64,
                                               num_regions=5,seq_length=5).to(self.device).float()


        else:
            raise ValueError(f'Invalid decoder type, expected one of '
                             f'[`lstm`, `mlp`], but got ``{self.decoder_name}``.')

    def encode(self, input) -> torch.Tensor:
        """
        draw a batch of samples from X, encode them to S and calculate
        the initial state ˆs0

        Parameters
        ----------
        input: Tensor
            a batch samples from X

        Returns
        -------
        out: Tensor
            encoder_output.shape=(batch_size, n_nodes, embed_dim)
        """

        self.encoder_output = self.encoder(input).float()

        return self.encoder_output
