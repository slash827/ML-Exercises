import sys
import matplotlib.pyplot as plt
import numpy as np


def read_info():
    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    centroids = np.loadtxt(centroids_fname)  # load centroids

    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.

    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels
    pixels = pixels.reshape(-1, 3)
    return centroids, pixels, out_fname


def average_all_points(centroids, pixels, tags):
    new_centroids = np.zeros(centroids.shape, dtype=float)
    samples_per_centroid = np.zeros(new_centroids.size//3, dtype=int)

    for pix_index, pixel in enumerate(pixels):
        cent_index = tags[pix_index]
        new_centroids[cent_index] = np.add(pixel, new_centroids[cent_index])
        samples_per_centroid[cent_index] += 1

    for cent_index, centroid in enumerate(new_centroids):
        if samples_per_centroid[cent_index] != 0:
            new_centroids[cent_index] /= samples_per_centroid[cent_index]
        else:
            print(centroids)
            print(new_centroids[cent_index])
            new_centroids[cent_index] = centroids[cent_index]
    return new_centroids


def calc_new_centroids(prev_centroids, pixels, tags):
    new_centroids = np.copy(prev_centroids)

    for pix_index, pixel in enumerate(pixels):
        min_dist = 2  # greater than maximum distance possible
        min_index = tags[pix_index]

        for cent_index, centroid in enumerate(new_centroids):
            sub = np.subtract(centroid, pixel)
            current_dist = np.linalg.norm(sub, 2) ** 2
            if current_dist < min_dist:
                min_dist = current_dist
                min_index = cent_index

        tags[pix_index] = min_index

    new_centroids = average_all_points(new_centroids, pixels, tags)
    new_centroids = new_centroids.round(4)
    changed = not np.array_equal(new_centroids, prev_centroids)
    return new_centroids, changed


def k_means(centroids, pixels):
    all_generations = []
    all_tags = []
    prev_centroids, new_centroids = centroids, []
    tags = np.zeros((pixels.size//3), dtype=int)
    iterations = 0
    while iterations < 20:
        new_centroids, changed = calc_new_centroids(prev_centroids, pixels, tags)
        all_generations.append(new_centroids)
        all_tags.append(tags)
        print(f'at iteration {iterations}: new cents are:\n{new_centroids}')

        iterations += 1
        if not changed:
            break
        prev_centroids = np.copy(new_centroids)

    return all_generations, iterations, all_tags


def output_file(all_generations, iterations, out_fname):
    with open(out_fname, 'w') as f:
        for iteration in range(iterations):
            output = f"[iter {iteration}]:{','.join([str(i) for i in all_generations[iteration]])}\n"
            f.write(output)


def main():
    centroids, pixels, out_fname = read_info()
    print(sys.argv)
    all_generations, iterations, all_tags = k_means(centroids, pixels)
    output_file(all_generations, iterations, out_fname)


def calculate_cost(centroids, pixels, tags):
    total_loss = 0
    for pix_index, pixel in enumerate(pixels):
        centroid = centroids[tags[pix_index]]
        sub = np.subtract(centroid, pixel)
        total_loss += np.linalg.norm(sub, 2) ** 2
    return total_loss / (pixels.size//3)


def plot_graph(centroids):
    pixels = read_info()[1]
    all_generations, iterations, all_tags = k_means(centroids, pixels)

    costs = []
    for i in range(len(all_generations)):
        costs.append(calculate_cost(all_generations[i], pixels, all_tags[i]))

    plt.plot(costs)
    plt.title("Cost Vs Iteration")
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.show()


def plot_different_graphs():
    for i in (2, 4, 8, 16):
        centroids = np.random.uniform(low=0, high=1, size=(i, 3))
        plot_graph(centroids)


if __name__ == '__main__':
    # plot_different_graphs()
    main()
