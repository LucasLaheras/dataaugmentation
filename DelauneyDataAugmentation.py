from ConvAutoEncoder import ConvAutoEncoder
from Lamp import Lamp
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial import Delaunay
from keras.datasets import mnist
from PIL import Image
import os
from Handler import Handler
from ConvAutoEncoder import configureDataset
import matplotlib.pyplot as plt
from scipy.spatial import distance


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [min(x_coordinates), min(y_coordinates)], [max(x_coordinates), max(y_coordinates)]


def calc_barycentric(pt, vert_a, vert_b, vert_c):
    pt, vert_a, vert_b, vert_c = pt.copy(), vert_a.copy(), vert_b.copy(), vert_c.copy()
    vert_a = np.append(vert_a, 1)
    vert_b = np.append(vert_b, 1)
    vert_c = np.append(vert_c, 1)
    pt = np.append(pt, 1)
    print(pt, vert_a, vert_b, vert_c)
    mat = np.array([vert_a, vert_b, vert_c])
    ans = np.linalg.solve(mat, pt)

    return ans


def configure_dataset():
    Handler().write_datafile()

    fist, last = 50, 150

    img, classify = Handler().read_datafile()

    img_test = np.asarray([img[i] for i in range(fist, last)])
    classfi_test = np.asarray([classify[i] for i in range(fist, last)])

    for i in range(last, fist-1, -1):
        np.delete(img, i)
        np.delete(classify, i)

    img, classify, img_test, classfi_test = img.astype('float32'), np.asarray(classify).astype('float32'), img_test.astype('float32'), classfi_test.astype('float32')
    return img, np.asarray(classify), img_test, classfi_test


def save_img(imgs, stri):
    num = imgs.shape[0]

    for i in range(num):
        A = imgs[i].copy()*255
        A = np.reshape(np.ravel(A), (52, 52))
        new_p = Image.fromarray(A)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(os.path.join(stri, str(i) + ".jpg"))


def greater_euclidean_distances(X_nd, m, threshold):
    x = np.copy(X_nd)
    ids = list()
    distances = np.zeros((X_nd.shape[0], X_nd.shape[0]))
    remove_list = list()
    for i in range(x.shape[0]):
        for j in range(X_nd.shape[0]):
            if i != j:
                distances[i][j] = distance.euclidean(X_nd[i], X_nd[j])
                if threshold >= distances[i][j] and not i in remove_list and not j in remove_list:
                        remove_list.append(i)

    array = np.copy(distances.reshape(-1))
    array = np.unique(array)
    array[::-1].sort()

    for i in range(x.shape[0]):
        for j in range(X_nd.shape[0]):
            for k in range(m*2):
                if array[k] == distances[i][j]:
                    ids.append(i)
                    ids.append(j)
    ids = np.unique(ids)

    x = np.delete(x, remove_list[:], 0)

    return ids, x

if __name__ == '__main__':
    #img_train, img_test = configureDataset()

    img_train, classify_train, img_test, classify_test = configure_dataset()

    print(img_train[0].shape)

    auto = ConvAutoEncoder(img_train[0].shape, img_train[0].shape, filters=[8, 8, 8])

    #auto.fit(img_train, img_train, epochs=500, batch_size=128, shuffle=True, validation_data=(img_test, img_test))

    #auto.save_weights(prefix="db_08-08-08_")

    auto.load_weights(prefix="db_08-08-08_")

    img_test_encoded = auto.encode(img_train)

    # Control Points 2D using TSNE Algorithm
    X_nd = np.reshape(img_test_encoded, (img_test_encoded.shape[0], int(np.prod(img_test_encoded.shape) / img_test_encoded.shape[0])))
    # Adding label
    X_nd = np.transpose(X_nd)
    X_nd = np.append(X_nd, [classify_train], axis=0)
    X_nd = np.transpose(X_nd)

    number_of_control_points = 10

    control_points_id, x_nd_removed = greater_euclidean_distances(X_nd, number_of_control_points, 30)
    number_of_control_points = control_points_id.shape[0]
    print(number_of_control_points)
    #control_points_id = np.random.randint(0, high=X_nd.shape[0], size=(number_of_control_points,))
    ctp_tsne = TSNE(n_components=2)
    ctp_proj = ctp_tsne.fit_transform(X_nd[control_points_id, 0:-1])
    ctp_proj = np.hstack((ctp_proj, control_points_id.reshape(number_of_control_points, 1)))

    # Lamp Algorithm
    lamp_proj = Lamp(Xdata=X_nd, control_points=ctp_proj, label=True, scale=True)
    X_proj = lamp_proj.fit()

    # Format data
    points1 = np.transpose([X_proj[:, 0], X_proj[:, 1], X_proj[:, -1]])
    for i in range(points1.shape[0]):
        points1[i][-1] = classify_train[i]
    points = np.compress([True, True], points1, axis=1)

    # New points using Gaussian distribution
    bb = np.array(bounding_box(points))

    number_of_new_points = 30

    length_of_diagonal = (np.linalg.norm(bb[0] - bb[1]))

    new_points = np.random.normal(points[0], length_of_diagonal, (number_of_new_points, 2))
    for i in range(1, int(points.size / 2)):
        new = np.random.normal(points[i], length_of_diagonal, (number_of_new_points, 2))
        new_points = np.concatenate((new_points, new), axis=0)

    # Verification, classification of each new point and location in the n dimensional space
    tri = Delaunay(points)

    classification = []
    new_points_nd = []

    for i in range(int(new_points.size / 2 - 1), -1, -1):

        triangle = tri.find_simplex(new_points[i])

        # point does not belong to a triangle or the points of the triangle do not have the same classification
        if (triangle == -1 or not (
                points1[tri.simplices[triangle]][0][-1] == points1[tri.simplices[triangle]][1][-1] and
                points1[tri.simplices[triangle]][1][-1] == points1[tri.simplices[triangle]][2][-1])):
            new_points = np.delete(new_points, i, 0)
        else:
            # new points valid in n dimentional sapce
            classification.insert(0, X_nd[tri.simplices[triangle]][0][-1])
            alpha, beta, gamma = calc_barycentric(new_points[i], points[tri.simplices[triangle]][0],
                                                  points[tri.simplices[triangle]][1],
                                                  points[tri.simplices[triangle]][2])
            x = alpha * X_nd[tri.simplices[triangle]][0] + beta * X_nd[tri.simplices[triangle]][1] + gamma * \
                X_nd[tri.simplices[triangle]][2]
            x[-1] = X_nd[tri.simplices[triangle]][0][-1]
            new_points_nd.append(x)

    plt.scatter(points1[:, 0], points1[:, 1], c=points1[:, -1])
    plt.scatter(ctp_proj[:, 0], ctp_proj[:, 1], c='m')
    plt.show()

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.show()

    plt.scatter(new_points[:, 0], new_points[:, 1], c='r')
    plt.show()

    # Format data
    new_points_nd = np.asarray(new_points_nd)
    classification_new_points = new_points_nd[:][-1]
    new_points_nd = np.delete(new_points_nd, -1, 1)
    print(new_points_nd.shape)
    deco_new_points = np.reshape(new_points_nd, (new_points_nd.shape[0], img_test_encoded.shape[1], img_test_encoded.shape[2], img_test_encoded.shape[3]))
    # deco_new_points = np.reshape(new_points_nd, (new_points_nd.shape[0], 4, 4, 8))
    print(deco_new_points.shape)
    new_points_decoded = auto.decode(img_test_encoded)
    testezinho = new_points_decoded[0]
    testezinho = np.reshape(testezinho, (52, 52))
    save_img(new_points_decoded, 'new_images/db/')
