import torch
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, input_shape: list, latent_space_dims: int):
        """
        The encoder network of the Two Phase Supervised Encoder.
        :param input_shape: the input shape in channels last format.
        :param latent_space_dims: the number of dimensions in the latent space.
        """
        super(Encoder, self).__init__()

        # save the input shape
        self.input_shape = input_shape

        # Convolutional Module
        self.enc_conv = torch.nn.Sequential(
            # =======================================================================
            torch.nn.Conv2d(in_channels=self.input_shape[-1],
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(2, 2),
                            padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout2d(0.3),
            torch.nn.ReLU(),
            # =======================================================================
            # =======================================================================
            torch.nn.Conv2d(in_channels=256,
                            out_channels=128,
                            kernel_size=(3, 3),
                            stride=(2, 2),
                            padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout2d(0.3),
            torch.nn.ReLU(),
            # =======================================================================
        )

        self.flatOutShape = (self.input_shape[0] // (2 ** 2)) * (self.input_shape[1] // (2 ** 2)) * 128

        # NN Module
        self.enc_nn = torch.nn.Sequential(
            torch.nn.Linear(self.flatOutShape, latent_space_dims)
        )

    def forward(self, x):
        # forward pass through the cnn network of the encoder
        x = self.enc_conv(x)

        # flatten
        x = torch.reshape(x, (x.shape[0], -1))

        # get the latent representations
        latent_repr = self.enc_nn(x)

        # return the latent representations
        return latent_repr


class Classifier(torch.nn.Module):

    def __init__(self, latent_space_dims: int,
                 number_of_classes: int):
        """
        The separator network of the Two Phase Supervised Encoder.
        :param latent_space_dims: the dimensionality of the latent space.
        :param number_of_classes: the number of classes.
        """

        super(Classifier, self).__init__()

        # the fully connected component of the separator network
        self.cl_nn = torch.nn.Sequential(
            torch.nn.Linear(latent_space_dims, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, number_of_classes)
        )

    def forward(self, x):
        # forward pass through the separator network and return the result
        return self.cl_nn(x)


class Decoder(torch.nn.Module):
    def __init__(self, input_shape: list, latent_space_dims: int):
        """
        The decoder network of the Two Phase Supervised Encoder.

        :param input_shape: the input shape in channels last format.
        :param latent_space_dims: the number of dimensions in the latent space.
        """
        super(Decoder, self).__init__()

        # save the input shape
        self.input_shape = input_shape

        # calculation of flattening 1d shape
        self.flatOutShape = (self.input_shape[0] // (2 ** 2)) * (self.input_shape[1] // (2 ** 2)) * 128

        # NN Module
        self.dec_nn = torch.nn.Sequential(
            torch.nn.Linear(latent_space_dims, self.flatOutShape),
            torch.nn.ReLU()
        )

        # Reshape Layer
        self.unflatten = torch.nn.Unflatten(dim=1,
                                            unflattened_size=(
                                                128, self.input_shape[0] // (2 ** 2), self.input_shape[1] // (2 ** 2)))

        # CNN module
        self.dec_conv = torch.nn.Sequential(
            # =======================================================================
            torch.nn.ConvTranspose2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # =======================================================================
            # =======================================================================

            torch.nn.ConvTranspose2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # =======================================================================
            # =======================================================================

            # output layer
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=self.input_shape[-1],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # pass through the fully connected network module of the decoder
        x = self.dec_nn(x)

        # reshape back to 3d
        x = self.unflatten(x)

        # pass through the convolutional network of the decoder
        reconstructions = self.dec_conv(x)
        return reconstructions


class TPSE(torch.nn.Module):
    def __init__(self, encoder, decoder, classifier):
        """
        The Two Phase Supervised Encoder.
        :param encoder: the encoder network.
        :param decoder: the decoder network.
        :param classifier: the separator network.
        """

        super(TPSE, self).__init__()

        # the encoder network
        self.encoder = encoder

        # the decoder network
        self.decoder = decoder

        # the separator network
        self.classifier = classifier

    def forward(self, x):
        # get the latent representations
        latent_repr = self.encoder(x)

        # assess their separability
        logits = self.classifier(latent_repr)

        # reconstruct images from their latent representations
        reconstructions = self.decoder(latent_repr)

        return latent_repr, logits, reconstructions


def train(dataloader, model, supervised_loss_fn,
          unsupervised_loss_fn, optim, device="cuda", report_value=100) -> None:
    """
    A function for training TP-SE utilizing the data loaded by the dataloader.

    :param dataloader: the dataloader.
    :param model: the TP-SE model.
    :param supervised_loss_fn: the supervised loss function.
    :param unsupervised_loss_fn: the unsupervised loss function.
    :param optim: the optimizer.
    :param device: the device to send a batch.
    :param report_value: report training loss values to screen per report_value batches.
    :return:
    None
    """

    # get the size of the data
    size = len(dataloader.dataset)

    # set the model to train mode
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # send the batch to the device
        X, y = X.to(device), y.to(device)

        # calculate the network outputs
        _, logits, reconstruction = model(X)

        # calculate loss functions
        supervised_loss = supervised_loss_fn(logits, y)
        unsupervised_loss = unsupervised_loss_fn(reconstruction, X)

        # back propagate the error in a two phase manner
        optim.zero_grad()
        unsupervised_loss.backward(retain_graph=True)  # unsupervised error
        supervised_loss.backward()  # supervised error
        optim.step()

        # report training loss values to screen per report_value batches.
        if batch % report_value == 0:
            current = batch * len(X)
            print(
                f"supervised loss {supervised_loss.item():>4f} unsupervised loss {unsupervised_loss.item():>4f}"
                f"            [{current:>5d}/{size:>5d}]")


def test(dataloader, model, supervised_loss_fn, unsupervised_loss_fn, device="cuda") -> float:
    """
    A function for evaluating TP-SE's performance utilizing the data loaded by the dataloader.

    :param dataloader: the dataloader.
    :param model: the TP-SE model.
    :param supervised_loss_fn: the supervised loss function.
    :param unsupervised_loss_fn: the unsupervised loss function.
    :param device: the device to send the batch.
    :return:
    The accuracy score.
    """

    # get the size of the data
    size = len(dataloader.dataset)

    # get the number of batches
    num_batches = len(dataloader)

    # set the model to evaluation mode
    model.eval()

    correct = 0
    test_sup_loss, test_unsup_loss = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # send the batch to the device
            X, y = X.to(device), y.to(device)

            # calculate the network outputs
            _, logits, reconstruction = model(X)

            # calculate loss function
            supervised_loss = supervised_loss_fn(logits, y)
            unsupervised_loss = unsupervised_loss_fn(reconstruction, X)

            test_sup_loss += supervised_loss.item()
            test_unsup_loss += unsupervised_loss.item()

            correct += (logits.argmax(1) == y).type(torch.float).sum().item()

    # compute the evaluation metrics
    test_sup_loss /= num_batches
    test_unsup_loss /= num_batches
    correct /= size

    # report the evaluation metrics to screen
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Supervised Loss {test_sup_loss:>4f}, Avg "
        f"Unsupervised Loss {test_unsup_loss:>4f}  \n")

    # return the accuracy score
    return correct


def get_inputs_outputs_latent_representations(dataloader,
                                              model,
                                              get_input: bool = False,
                                              get_gt: bool = False,
                                              get_lr: bool = False,
                                              get_reconstructions: bool = False,
                                              get_logits: bool = False,
                                              device="cuda"):
    return_tuple = tuple()

    if get_input:
        input_data = list()
    if get_gt:
        ground_truth = list()
    if get_reconstructions:
        reconstructions = list()
    if get_logits:
        logits = list()
    if get_lr:
        latent_representations = list()

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            # send the batch to the gpu
            X_device = X.to(device)

            # calculate the network outputs
            batch_latent_representations, batch_logits, batch_reconstructions = model(X_device)

            if get_input:
                input_data.append(X.numpy())
            if get_gt:
                ground_truth.append(y.numpy())
            if get_reconstructions:
                reconstructions.append(batch_reconstructions.to("cpu").detach().numpy())
            if get_logits:
                logits.append(batch_logits.to("cpu").detach().numpy())
            if get_lr:
                latent_representations.append(batch_latent_representations.to("cpu").detach().numpy())

    if get_input:
        input_data = np.concatenate(input_data, axis=0)
        return_tuple = return_tuple + (input_data,)
    if get_gt:
        ground_truth = np.concatenate(ground_truth, axis=0).astype("str")
        return_tuple = return_tuple + (ground_truth,)
    if get_reconstructions:
        reconstructions = np.concatenate(reconstructions, axis=0)
        return_tuple = return_tuple + (reconstructions,)
    if get_logits:
        logits = np.concatenate(logits, axis=0)
        return_tuple = return_tuple + (logits,)
    if get_lr:
        latent_representations = np.concatenate(latent_representations, axis=0)
        return_tuple = return_tuple + (latent_representations,)

    # order: (input, ground_truth, reconstructions, logits, latent_representations)
    return return_tuple
