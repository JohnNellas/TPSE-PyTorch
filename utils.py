import numpy as np
import torch
import matplotlib.offsetbox
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics
import os
import argparse


def decision_boundary_on_latent_space(classifier,
                                      embeddings: np.ndarray,
                                      ground_truth: np.ndarray,
                                      zoom: float = 0.5,
                                      figsize: tuple = (12, 10),
                                      save_flag: bool = True,
                                      save_path: str = "output_fig.jpg",
                                      number_of_grid_points: int = 100,
                                      number_of_samples_per_class: int = 5,
                                      display_images_flag: bool = True,
                                      images_sample_pool: np.ndarray = None,
                                      samples_to_display: np.ndarray = None,
                                      cmap: str = "gray",
                                      seed_val: int = 42,
                                      label_mapping: dict = None,
                                      display_fig_flag: bool = False,
                                      dpi: int = 300,
                                      device: str = "cuda") -> None:
    """
    A function for generating a scatter plot of the latent representations (embeddings)
    where the decision boundaries and regions of the separator network (classifier)
    are drawn as background, while concurrently some embeddings can be replaced by
    their original image (images_sample_pool).

    :param classifier: the separator network.
    :param embeddings: the latent representations to be displayed.
    :param ground_truth: the ground truth of the provided latent representations.
    :param zoom: the zoom imposed on the displayed images.
    :param figsize: the figure size as tuple (width, height) in inches.
    :param save_flag: a flag for saving the final figure.
    :param save_path: target destination to save the generated figure.
    :param number_of_grid_points: the number of points to generate in each dimension.
    :param number_of_samples_per_class: the number of samples per class to be replaced by their original image.
    :param display_images_flag: display images on the scatter plot if desired.
    :param images_sample_pool: the original images of each latent representation.
    :param samples_to_display: specific embedding indices to be replaced by their original image
    (otherwise random).
    :param cmap: the colormap of images.
    :param seed_val: the seed value.
    :param label_mapping: the label mapping.
    :param display_fig_flag: a flag for displaying the figure on screen.
    :param dpi: the employed dpi when saving the final figure.
    :param device: the utilized device.
    :return:
    None
    """

    # check if the embeddings are 2D
    if embeddings.shape[-1] != 2:
        print("Please provide 2D embeddings")
        return

    # create a grid around the embeddings
    min_x = np.round(np.min(embeddings[:, 0]), 2)
    max_x = np.round(np.max(embeddings[:, 0]), 2)

    min_y = np.round(np.min(embeddings[:, 1]), 2)
    max_y = np.round(np.max(embeddings[:, 1]), 2)

    x = np.linspace(min_x, max_x, number_of_grid_points)
    y = np.linspace(min_y, max_y, number_of_grid_points)

    X, Y = np.meshgrid(x, y)

    # create a list of all the 2d grid points
    grid_of_points = np.dstack((X, Y)).reshape((-1, 2)).astype("float32")

    # convert to a torch tensor of float32 (in order to be float not double - float64 - )
    grid_of_points_torch = torch.from_numpy(grid_of_points).to(device)

    # predict their class with the isolated classifier network
    with torch.no_grad():
        logits = classifier(grid_of_points_torch)

    predictions_grid = np.argmax(logits.to("cpu").detach().numpy(), axis=1)

    # do the label mapping if specified
    if label_mapping:
        predictions_grid = [label_mapping[nid] for nid in predictions_grid]
        predictions_grid = np.array(predictions_grid)

    # create the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Do a scatter plot of the classified grid points coloured by the predicted class
    df = pd.DataFrame(data=grid_of_points, columns=["E1", "E2"])
    df["Y"] = predictions_grid
    df["Y"] = df["Y"].astype('category')

    sns.scatterplot(x="E1", y="E2", hue="Y",
                    data=df, palette="bright",
                    alpha=0.25, marker="s", legend=False)

    # Do a scatter plot of the raw embeddings coloured by the ground truth class
    df_sc = pd.DataFrame(data=embeddings, columns=["E1", "E2"])
    df_sc["Y"] = ground_truth
    df_sc["Y"] = df_sc["Y"].astype('category')
    sns.scatterplot(x="E1", y="E2", hue="Y", data=df_sc, palette="bright")

    # if not specific samples are specified to be displayed, sample number_of_samples_per_class samples per class
    # to be displayed
    if samples_to_display is None:

        # get a random number of samples from each class
        np.random.seed(seed_val)
        samples_each_class = list()
        for uq_class in np.unique(ground_truth):
            inds_class = np.where(ground_truth == uq_class)[0]
            samples_of_class = np.random.choice(inds_class, size=(number_of_samples_per_class,), replace=False)
            samples_each_class.append(samples_of_class)
        samples_to_display = np.concatenate(samples_each_class, axis=0)

    # if specified, replace the samples_to_display embeddings by their corresponding original image
    if display_images_flag:
        # replace the sampled points by the corresponding original images in the scatter plot
        for sample in samples_to_display:
            im = matplotlib.offsetbox.OffsetImage(images_sample_pool[sample].squeeze(), zoom=zoom, cmap=cmap)
            x0, y0 = embeddings[sample]
            ab = matplotlib.offsetbox.AnnotationBbox(im, (x0, y0), frameon=False)
            ax.add_artist(ab)

    # disable the axis
    plt.axis("off")

    if save_flag:
        # save it to a file
        plt.savefig(save_path,
                    dpi=dpi,
                    bbox_inches='tight')

    if display_fig_flag:
        plt.show()
    else:
        plt.close()


def evaluate_method(truth: np.ndarray,
                    pred: np.ndarray,
                    save_flag: bool = True,
                    filename: str = "evaluation.txt") -> list:
    """
    A function for evaluating a classification method using the
    classification_report function (sklearn.metrics) utilizing the provided ground truth and predicted values.
    :param truth: the ground truth.
    :param pred: the predictions.
    :param save_flag: a flag for saving the obtained classification report.
    :param filename: the utilized filename when saving.
    :return: a list containing the classification report in string and dictionary format along with the analytic string
    displaying Accuracy and weighted average of F1-Score.
    """
    # acquire the classification report in string and dictionary format along with the analytic string
    cls_rep_str = sklearn.metrics.classification_report(truth, pred)
    cls_rep_dict = sklearn.metrics.classification_report(truth, pred, output_dict=True)
    analytic_string = f"Analytically, Accuracy: {cls_rep_dict['accuracy']}," + \
                      f" weighted_avg_f1: {cls_rep_dict['weighted avg']['f1-score']}"

    # save the results if specified
    if save_flag:
        with open(filename, "w") as f:
            print(f"{cls_rep_str} \n\n{analytic_string}", file=f)

    # return the classification report in string and dictionary format along with the analytic string
    return [cls_rep_str, cls_rep_dict, analytic_string]


def evaluate_with_classifier(classifier_object,
                             x_tr: np.ndarray,
                             y_tr: np.ndarray,
                             x_te: np.ndarray,
                             y_te: np.ndarray,
                             save_flag: bool = True,
                             filename: str = "evaluation.txt") -> list:
    """
    A function for training and evaluating a classifier model (scikit-learn classifier object)
    :param classifier_object: the scikit-learn classifier object.
    :param x_tr: the training data.
    :param y_tr: the ground truth of the training data.
    :param x_te: the testing data.
    :param y_te: the ground truth of the testing data.
    :param save_flag: a flag for saving the obtained classification report.
    :param filename: the utilized filename when saving.
    :return: a list containing the classification report in string and dictionary format along with the analytic string
    displaying Accuracy and weighted average of F1-Score.
    """

    # train the provided classifier utilizing the training dataset
    classifier_object.fit(x_tr, y_tr)

    # acquire the predictions from the trained classifier utilizing the test dataset
    preds = classifier_object.predict(x_te)

    # evaluate the results
    cls_rep_str, cls_rep_dict, analytic_string = evaluate_method(y_te, preds, save_flag, filename)

    # return the classification report in string and dictionary format along with the analytic string
    return [cls_rep_str, cls_rep_dict, analytic_string]


def find_point_in_radius(selected_point: np.ndarray,
                         list_of_points: np.ndarray,
                         radius: float):
    """
    A function for finding if a point is inside the radius of existing points.
    :param selected_point: the selected candidate point.
    :param list_of_points: the list containing the existing points.
    :param radius: the radius of each existing point.
    :return: the index of the first point fulfilling the condition in the list of points otherwise -1.
    """

    idx_point_in_radius = -1

    for idx, point in enumerate(list_of_points):

        # calculate the distance of the candidate point from the existing point
        norm = np.linalg.norm(point - selected_point)

        # if inside the radius, point is found break the loop
        if norm < radius:
            idx_point_in_radius = idx
            break

    # return the index of the first discovered point, otherwise return -1
    return idx_point_in_radius


def latent_space_exploration_image_generation(classifier,
                                              decoder,
                                              xbounds: tuple,
                                              ybounds: tuple,
                                              zoom: float = 0.5,
                                              figsize: tuple = (12, 8),
                                              cmap: str = "gray",
                                              save_directory="generated_figures",
                                              output_format="jpg",
                                              number_of_grid_points: int = 150,
                                              device: str = None,
                                              mapping: dict = None,
                                              radius: float = 1.2) -> None:
    """
    A function for interactive 2D latent space exploration and image generation.
    Controls:
    b -> decode latent representation under cursor.
    n -> display already decoded image under the cursor.
    m -> save the already decoded image under the cursor.

    :param classifier: the separator network.
    :param decoder: the decoder network.
    :param xbounds: the min and max values for the x-axis.
    :param ybounds: the min and max values for the y-axis.
    :param zoom: the zoom of the decoded images.
    :param figsize: the figure size as a tuple (width, height) in inches.
    :param cmap: the colormap of the displayed images.
    :param save_directory: the target directory when saving the decoded images.
    :param output_format: the format of the decoded images when saved.
    :param number_of_grid_points: the number of points sampled between the boundaries in x and y-axis.
    :param device: the device to send the data.
    :param mapping: the label mapping.
    :param radius: the radius of a selected point (utilized for checking if the mouse is over a decoded image).
    :return: None.
    """

    # create a grid around the specified bounding box
    min_x, max_x = xbounds
    min_y, max_y = ybounds

    x = np.linspace(min_x, max_x, number_of_grid_points)
    y = np.linspace(min_y, max_y, number_of_grid_points)

    X, Y = np.meshgrid(x, y)

    # create a list of all the 2d grid points
    grid_of_points = np.dstack((X, Y)).reshape((-1, 2)).astype("float32")

    # convert to a torch tensor of float32 (in order to be float not double - float64 - )
    grid_of_points_torch = torch.from_numpy(grid_of_points).to(device)

    # predict their class with the isolated separator network
    with torch.no_grad():
        logits = classifier(grid_of_points_torch)

    predictions_grid = np.argmax(logits.to("cpu").detach().numpy(), axis=1)

    # do the label mapping if specified
    if mapping:
        predictions_grid = [mapping[nid] for nid in predictions_grid]
        predictions_grid = np.array(predictions_grid)

    # Do a scatter plot of the classified grid points coloured by the predicted class
    # and set it in the left frame
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    df = pd.DataFrame(data=grid_of_points, columns=["E1", "E2"])

    df["Pr. Class"] = predictions_grid
    df["Pr. Class"] = df["Pr. Class"].astype('category')

    sns.scatterplot(x="E1", y="E2", hue="Pr. Class", ax=ax[0],
                    data=df, palette="bright",
                    alpha=0.25, marker="s", legend=True)

    ax[0].set_title("Latent Space")
    sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))

    # create a black image and set it in the right frame
    black_image = np.zeros((28, 28))
    ax[1].imshow(black_image, cmap=cmap)

    # mutatable objects for the nested function
    selected_points = list()
    selected_decoded_image = list()

    def on_click(event):

        if event.key == "b":
            # if b is pressed (decode point)

            # get the selected point
            selected_point = np.array(
                [[event.xdata, event.ydata]], dtype="float32")

            # save it to the list of selected points
            selected_points.append(selected_point)

            # decode the image from the selected latent representation
            selected_point_tensor = torch.from_numpy(selected_point).to(device)
            with torch.no_grad():
                decoded_image = decoder(selected_point_tensor)
            decoded_image = decoded_image.squeeze()

            # if the image is an rgb image convert to channels last format
            if decoded_image.shape[0] == 3:
                decoded_image = torch.moveaxis(decoded_image, 0, 2)

            decoded_image = decoded_image.to("cpu").detach().numpy().squeeze()

            # save the decoded image to the list of decoded images
            selected_decoded_image.append(decoded_image)

            # display the decoded image on the specified position on the scatter plot
            im = matplotlib.offsetbox.OffsetImage(
                decoded_image, zoom=zoom, cmap=cmap)
            ab = matplotlib.offsetbox.AnnotationBbox(
                im, tuple(selected_point[0]), frameon=False)
            event.inaxes.add_artist(ab)

            # display the decoded image in the right panel
            ax[1].imshow(decoded_image, cmap=cmap)
            ax[1].set_title("Decoded Image")
            ax[1].axis("off")

            # update canvas
            fig.canvas.draw()

        elif event.key in ["m", "n"]:

            # if m or n button is pressed, namely display or save previously decoded image
            # when mouse is on top of it respectively

            # get the selected point
            clicked_point = np.array(
                [event.xdata, event.ydata], dtype="float32")

            # check if the selected point is in radius of previously selected points
            # and get its index
            idx_point_in_radius = find_point_in_radius(
                clicked_point, selected_points, radius)

            # if the point is inside the radius of a previously selected point
            if idx_point_in_radius != -1:
                # get the decoded image
                image_in_radius = selected_decoded_image[idx_point_in_radius]
            else:
                # else get a black image
                image_in_radius = black_image

            if event.key == "n":
                # if display is specified

                # display the image on the right panel
                ax[1].imshow(image_in_radius, cmap=cmap)
                ax[1].set_title("Decoded Image")
                ax[1].axis("off")

                # update canvas
                fig.canvas.draw()
            elif event.key == "m":
                # if save is specified

                # if the target directory does not exist then create it
                if not os.path.isdir(save_directory):
                    os.mkdir(save_directory)

                # save the image
                save_path = os.path.join(
                    save_directory, f"decoded_image_{on_click.save_cntr}.{output_format}")
                print(f"saved at {save_path}")
                on_click.save_cntr = on_click.save_cntr + 1
                plt.imsave(save_path, image_in_radius, cmap=cmap)

    # counter for the saved images
    on_click.save_cntr = 1

    # connect the event for key presses
    cid = fig.canvas.mpl_connect('key_press_event', on_click)

    # disable the axis and set x and y limits
    ax[0].axis("off")
    ax[1].axis("off")
    ax[1].set_title("Decoded Image")
    ax[0].set_xlim((min_x, max_x))
    ax[0].set_ylim((min_y, max_y))

    # set a tight layout
    plt.tight_layout()

    # display the figure
    plt.show()


def non_negative_int_input(value):
    """
    A function for checking if an input value is a non-negative integer.
    :param value: the input value.
    :return: the non-negative integer value if this holds otherwise raises an exception.
    """

    try:
        # try to convert input value to integer
        value = int(value)

        # if conversion is successful check if the integer is non-negative
        if value < 0:
            # raise an exception if the integer is not a non-negative integer
            raise argparse.ArgumentTypeError(f"{value} is not a non-negative integer")
    except ValueError:

        # if conversion to integer fails then the input is not an integer
        raise argparse.ArgumentTypeError(f"{value} is not an integer.")

    # return the non-negative integer value if every process is successfully completed
    return value


def non_negative_float_input(value):
    """
    A function for checking if an input value is a non-negative float.
    :param value: the input value.
    :return: the non-negative float if this holds otherwise raises an exception.
    """

    try:
        # try to convert input value to float
        value = float(value)

        # if conversion is successful check if the float is non-negative
        if value < 0:
            # raise an exception if the float is not a non-negative float
            raise argparse.ArgumentTypeError(f"{value} is not a non-negative float")
    except ValueError:

        # if conversion to float fails then the input is not a float
        raise argparse.ArgumentTypeError(f"{value} is not a float.")

    # return the non-negative float value if every process is successfully completed
    return value
