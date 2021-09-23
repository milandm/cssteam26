# Reference: https://github.com/miloharper/visualise-neural-network

from math import cos, sin, atan

import numpy as np
from matplotlib import pyplot

from modules.saferegion import SafeRegion1d


class Neuron:
    def __init__(self, x, y, neuron_radius, value):
        self.x = x
        self.y = y
        self.neuron_radius = neuron_radius
        self.value = value

    def draw(self):
        color = 'g' if self.value else 'r'
        circle = pyplot.Circle((self.x, self.y), radius=self.neuron_radius, fill=True, color=color)
        pyplot.gca().add_patch(circle)


class Layer:
    def __init__(self,
                 network,
                 number_of_neurons,
                 values,
                 neuron_radius,
                 horizontal_distance_between_neurons,
                 vertical_distance_between_layers,
                 number_of_neurons_in_widest_layer):
        self.neuron_radius = neuron_radius
        self.values = values
        self.horizontal_distance_between_neurons = horizontal_distance_between_neurons
        self.vertical_distance_between_layers = vertical_distance_between_layers
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y, self.neuron_radius, self.values[iteration])
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=0.01)
        pyplot.gca().add_line(line)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)


class NeuralNetwork:
    def __init__(self,
                 neuron_radius,
                 horizontal_distance_between_neurons,
                 vertical_distance_between_layers,
                 number_of_neurons_in_widest_layer):
        self.layers = []
        self.neuron_radius = neuron_radius
        self.horizontal_distance_between_neurons = horizontal_distance_between_neurons
        self.vertical_distance_between_layers = vertical_distance_between_layers
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer

    def add_layer(self, number_of_neurons, values):
        layer = Layer(self,
                      number_of_neurons,
                      values,
                      self.neuron_radius,
                      self.horizontal_distance_between_neurons,
                      self.vertical_distance_between_layers,
                      self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.axis('off')
        red_circle = pyplot.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=15)
        green_circle = pyplot.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=15)
        pyplot.legend([green_circle, red_circle], ['inside safe region', 'outside safe region'], loc="upper left")
        # pyplot.show()


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def visualize_in_out(model, draw_connections=False):
    number_of_neurons_in_widest_layer = 0
    safe_regions = dict()
    for name, layer in model.named_modules():
        if isinstance(layer, SafeRegion1d):
            # print("{}: {}".format(name, torch.sum(layer.in_out)))
            safe_regions[name] = (layer.num_features, layer.in_out[0].cpu().tolist())
            if layer.num_features > number_of_neurons_in_widest_layer:
                number_of_neurons_in_widest_layer = layer.num_features

    # full screen
    # mng = pyplot.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    # set figure height and width in inches
    dpi = 100
    width = 2000
    height = 2000
    num_layers = len(safe_regions)
    f = pyplot.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # draw network
    vertical_distance_between_layers = height / num_layers
    horizontal_distance_between_neurons = width / number_of_neurons_in_widest_layer
    neuron_radius = horizontal_distance_between_neurons / 3
    network = NeuralNetwork(neuron_radius,
                            horizontal_distance_between_neurons,
                            vertical_distance_between_layers,
                            number_of_neurons_in_widest_layer)

    for name, sr in safe_regions.items():
        network.add_layer(sr[0], sr[1])
    network.draw()

    image = fig2data(f)
    f.clear()
    pyplot.close(f)
    return image

# if __name__ == "__main__":
#     vertical_distance_between_layers = 6
#     horizontal_distance_between_neurons = 2
#     neuron_radius = 0.5
#     number_of_neurons_in_widest_layer = 64
#     network = NeuralNetwork()
#     network.add_layer(64)
#     network.add_layer(64)
#     network.add_layer(1)
#     network.draw()
