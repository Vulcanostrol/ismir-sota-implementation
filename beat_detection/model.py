"""
Author: Koen van den Brink
"""
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Input, Flatten, TimeDistributed, Conv2D,\
    MaxPooling2D, Dropout, Conv1D, SpatialDropout1D, Add, Concatenate
from tensorflow.keras import Model
from typing import List


def get_model(conv_dropout_rate: float = 0.15,
              no_tcn_layers: int = 11,
              filter_count: int = 16,
              kernel_size: int = 5,
              activation_func: str = 'elu',
              dilation_base: float = 2.0,
              no_conv_layers: int = 3,
              dilation_rates: List['float'] = None,
              spatial_dropout_rate: float = 0.1,
              transposed_spectrograms: bool = False):
    """
    Creates the state of the art (as of ISMIR 2020) beat detection/downbeat detection/tempo estimation machine learning
    model in TensorFlow. See documentation of individual model blocks to see the papers that were used to implement
    them.

    Keeping all parameters as default will use the parameters used in the state of the art papers and result in the
    model that is currently the best model known.

    :param conv_dropout_rate: The dropout rates in the convolutional block.
    :param no_tcn_layers: Number of TCN layers in the TCN block.
    :param filter_count: Amount of filters in both the convolutional- and TCN block.
    :param kernel_size: Kernel size for the 1D convolutional layers in the TCN layers.
    :param activation_func: Activation function for TCN layer.
    :param dilation_base: TCN layer dilation base rate. The dilations are exponential, and this is the exponentiation
    base.
    :param no_conv_layers: Number of convolutional layers in the TCN layers.
    :param dilation_rates: In case you do not want the TCN dilations to be exponential, you can specify specific
    dilation rates with this parameter as a list of dilation rates.
    :param spatial_dropout_rate: Spatial dropout rate for TCN layers.
    :param transposed_spectrograms: Whether the input spectrograms are transposed. In some setups, this can make data
    storage easier or even the training faster (for some odd reason). However, I would really recommend just keeping
    this False, unless you know EXACTLY what you're doing.
    :return:
    """

    # First dimension is None, because we are taking in any number of spectrograms.
    if transposed_spectrograms:
        _inputs = Input((None, 5, 81, 1), name='mel-spectrogram')
    else:
        # Default
        _inputs = Input((None, 81, 5, 1), name='mel-spectrogram')

    # First, the convolutional block.
    _convolutional_block = ConvolutionalBlock(filter_count=filter_count, dropout_rate=conv_dropout_rate)
    _output = _convolutional_block(_inputs)

    # Then the TCN block.
    _tcn_block = TCNBlock(no_tcn_layers=no_tcn_layers, filter_count=filter_count, kernel_size=kernel_size,
                          activation_func=activation_func, dilation_base=dilation_base, no_conv_layers=no_conv_layers,
                          dilation_rates=dilation_rates, spatial_dropout_rate=spatial_dropout_rate)
    _output, _skip_connections = _tcn_block(_output)

    # Then the beat and downbeat outputs.
    _x_beat = TimeDistributed(Dropout(0.15))(_output)
    _output_beat = TimeDistributed(Dense(1, 'sigmoid'), name='beat-y')(_x_beat)

    _x_downbeat = TimeDistributed(Dropout(0.15))(_output)
    _output_downbeat = TimeDistributed(Dense(1, 'sigmoid'), name='downbeat-y')(_x_downbeat)

    # Lastly, the tempo output.
    _skip_output = Add()(_skip_connections)
    _skip_output = Dropout(0.15)(_skip_output)
    _skip_output = GlobalAveragePooling1D()(_skip_output)
    _skip_output = Dropout(0.15)(_skip_output)
    _output_tempo = Dense(300, activation='softmax', name='tempo-y')(_skip_output)

    # Now, we create a model with the right input and outputs and return it.
    _model = Model(inputs=[_inputs], outputs=[_output_beat, _output_downbeat, _output_tempo])

    return _model


class ConvolutionalBlock(Model):
    def __init__(self,
                 filter_count: int = 16,
                 dropout_rate: float = 0.15,
                 transposed_input: bool = False):
        """
        As seen in the following paper:
        https://www.mdpi.com/2079-9292/10/13/1518
        """
        super(ConvolutionalBlock, self).__init__()

        self.__transposed_input = transposed_input

        if self.__transposed_input:
            self.__transpose_params = {
                "pooling__pool_size": (1, 3),
                "conv2d_filter_size": (1, 10)
            }
        else:
            # Default
            self.__transpose_params = {
                "pooling__pool_size": (3, 1),
                "conv2d_filter_size": (10, 1)
            }

        self.__conv2d_a = TimeDistributed(
            Conv2D(filter_count, (3, 3), activation='elu'))
        self.__max_pool_a = TimeDistributed(
            MaxPooling2D(pool_size=self.__transpose_params["pooling__pool_size"]))
        self.__dropout_a = TimeDistributed(
            Dropout(dropout_rate))

        # self.__conv2d_b = TimeDistributed(
        #     Conv2D(filter_count, (1, 10), activation='elu'))
        self.__conv2d_b = TimeDistributed(
            Conv2D(filter_count, self.__transpose_params["conv2d_filter_size"], activation='elu'))
        self.__max_pool_b = TimeDistributed(
            MaxPooling2D(pool_size=self.__transpose_params["pooling__pool_size"]))
        self.__dropout_b = TimeDistributed(
            Dropout(dropout_rate))

        self.__conv2d_c = TimeDistributed(
            Conv2D(filter_count, (3, 3), activation='elu'))
        # self.__max_pool_c = TimeDistributed(
        #     MaxPooling2D(pool_size=(1, 3)))
        self.__max_pool_c = TimeDistributed(
            MaxPooling2D(pool_size=self.__transpose_params["pooling__pool_size"]))
        self.__dropout_c = TimeDistributed(
            Dropout(dropout_rate))

        self.__flatten = TimeDistributed(
            Flatten())

    def call(self, inputs, training=None, mask=None):
        _outputs = self.__conv2d_a(inputs)
        _outputs = self.__max_pool_a(_outputs)
        _outputs = self.__dropout_a(_outputs)
        _outputs = self.__conv2d_b(_outputs)
        _outputs = self.__max_pool_b(_outputs)
        _outputs = self.__dropout_b(_outputs)
        _outputs = self.__conv2d_c(_outputs)
        _outputs = self.__max_pool_c(_outputs)
        _outputs = self.__dropout_c(_outputs)
        _outputs = self.__flatten(_outputs)

        return _outputs

    def get_config(self):
        return {}


class TCNBlock(Model):
    def __init__(self,
                 no_tcn_layers: int = 11,
                 filter_count: int = 16,
                 kernel_size: int = 5,
                 activation_func: str = 'elu',
                 dilation_base: float = 2.0,
                 no_conv_layers: int = 3,
                 dilation_rates: List['float'] = None,
                 spatial_dropout_rate: float = 0.1):
        super(TCNBlock, self).__init__()

        self.__no_tcn_layers = no_tcn_layers
        self.__filter_count = filter_count
        self.__kernel_size = kernel_size
        self.__activation_func = activation_func
        self.__dilation_base = dilation_base
        self.__no_conv_layers = no_conv_layers
        self.__dilation_rates = dilation_rates
        self.__spatial_dropout_rate = spatial_dropout_rate

        self.__tcn_layers = [
            TCNLayer(filter_count=self.__filter_count,
                     kernel_size=self.__kernel_size,
                     activation_func=self.__activation_func,
                     dilation_base=self.__dilation_base,
                     no_conv_layers=self.__no_conv_layers,
                     dilation_rates=self.__dilation_rates,
                     spatial_dropout_rate=self.__spatial_dropout_rate)
            for _ in range(self.__no_tcn_layers)
        ]

    def call(self, inputs, training=False, *args, **kwargs):
        """
        :return: The block's two outputs as a 2-tuple. First, the output of the last TCN layer and then a list of all
        the skip connection outputs.
        """
        _skip_connection_outputs = []
        _last_layer_output = inputs

        if len(self.__tcn_layers) == 0:
            raise ValueError("TCN Block has no layers. Please try a different configuration.")

        for _layer in self.__tcn_layers:
            _last_layer_output, _layer_skip = _layer(_last_layer_output, training=training, *args, **kwargs)
            _skip_connection_outputs.append(_layer_skip)

        return _last_layer_output, _skip_connection_outputs

    def get_config(self):
        return {
            "no_tcn_layers": self.__no_tcn_layers,
            "filter_count": self.__filter_count,
            "kernel_size": self.__kernel_size,
            "activation_func": self.__activation_func,
            "dilation_base": self.__dilation_base,
            "no_conv_layers": self.__no_conv_layers,
            "dilation_rates": self.__dilation_rates,
            "spatial_dropout_rate": self.__spatial_dropout_rate,
        }


class TCNLayer(Model):
    def __init__(self,
                 filter_count: int = 16,
                 kernel_size: int = 5,
                 activation_func: str = 'elu',
                 dilation_base: float = 2.0,
                 no_conv_layers: int = 3,
                 dilation_rates: List['float'] = None,
                 spatial_dropout_rate: float = 0.1):
        """
        A TCN layer in Keras. Based on the following papers (by Sebastian Böck)
        - https://www.ismir2020.net/assets/img/proceedings/2020_ISMIR_Proceedings.pdf (page 602-610)
        - http://telecom.inesctec.pt/~mdavies/pdfs/DaviesBoeck19-eusipco.pdf
        - https://www.mdpi.com/2079-9292/10/13/1518

                                                    TCN LAYER

                ┌───────────────────────────────────────» bypass ──┐
                │                                                  │
        ────────┼───» conv A ──┐                                   │
                │              ├───» ELU ──» dropout ──» skip ──» add ──────» next layer / output
                └───» conv B ──┘                                   │
                                                                outputs

        The layer is highly parameterizable, and has the following parameters that can be adjusted:

        :param filter_count: The number of convolutional filters in the TCN layer.
        :param kernel_size: The filter size/shape of the convolutional layers.
        :param activation_func: The activation function of the convolutional layers.
        :param dilation_base: The exponential base of dilation rate of the layers. The dilation rate of layer L (index
        starting at 0), when the dilation base is B, is B**L.
        :param no_conv_layers: The number of convolutional layers within the TCN layer.
        :param dilation_rates: A list containing the dilation rates of the layers. This will overwrite dilation_base and
        no_conv_layers if not None.
        :param spatial_dropout_rate: The spatial dropout rate of the TCN layer.
        """
        super(TCNLayer, self).__init__()

        self.__filter_count = filter_count
        self.__kernel_size = kernel_size
        self.__activation_func = activation_func
        self.__dilation_base = dilation_base
        self.__no_conv_layers = no_conv_layers
        self.__spatial_dropout_rate = spatial_dropout_rate

        if dilation_rates is None:
            self.__dilation_rates = [dilation_base ** i for i in range(self.__no_conv_layers)]
        else:
            self.__dilation_rates = dilation_rates

        self.__convolutional_layers = [Conv1D(
            filters=self.__filter_count,
            kernel_size=self.__kernel_size,
            dilation_rate=int(_dilation_rate),
            activation=self.__activation_func,
            padding='same'  # This is always 'same', and is not supposed to be parameterized according to the papers.
        ) for _dilation_rate in self.__dilation_rates]

        self.__concatenate = Concatenate()
        self.__spatial_dropout = SpatialDropout1D(spatial_dropout_rate)
        self.__skip = Conv1D(filters=self.__filter_count, kernel_size=1)
        self.__bypass = Conv1D(filters=self.__filter_count, kernel_size=1)
        self.__add = Add()

    def call(self, input_tensor, training=False, *args, **kwargs):

        # Calculate dilated convolutional outputs.
        _x_dilations = [
            _convolutional_layer(
                input_tensor,
                training=training,
                *args,
                **kwargs
            ) for _convolutional_layer in self.__convolutional_layers
        ]

        # Concatenate outputs.
        _x_concatenated = self.__concatenate(_x_dilations, training=training, *args, **kwargs)

        # Concatenated output goes through dropout.
        _x_dropped = self.__spatial_dropout(_x_concatenated, training=training, *args, **kwargs)

        # Skip output.
        _x_skip = self.__skip(_x_dropped, training=training, *args, **kwargs)

        # Bypassed output.
        _x_bypassed = self.__bypass(input_tensor, training=training, *args, **kwargs)

        # Added output.
        _x_added = self.__add([_x_skip, _x_bypassed], training=training, *args, **kwargs)

        return _x_added, _x_skip

    def get_config(self):
        return {
            "filter_count": self.__filter_count,
            "kernel_size": self.__kernel_size,
            "activation_func": self.__activation_func,
            "dilation_base": self.__dilation_base,
            "no_conv_layers": self.__no_conv_layers,
            "dilation_rates": self.__dilation_rates,
            "spatial_dropout_rate": self.__spatial_dropout_rate,
        }
