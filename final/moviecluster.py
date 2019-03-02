import numpy as np
import pulp
import sys
import warnings

users_max_id = 6040
movie_max_id = 3952

def print_bad_movie(movieid, ratings_num):
    print("Movie "+ str(movieid)+" ignored because it has only "+str(ratings_num)+" ratings", file=sys.stderr)
    return

def load_and_clear_subset(movie_subset_path):
    correlations_data = np.fromfile(str(sys.argv[0][:-15])+'correlations_data_set_saved', dtype=float)
    correlations_data = correlations_data.reshape(movie_max_id + 1, movie_max_id + 1)
    correlations_indicator = np.load(str(sys.argv[0][:-15])+'correlations_indicator_set_saved.npy')
    correlations_indicator = correlations_indicator.reshape(movie_max_id + 1, movie_max_id + 1)
    movie_name = np.load(str(sys.argv[0][:-15])+'movies_names.npy')
    movie_name = movie_name.reshape(movie_max_id + 1)
    bad_movies = np.load(str(sys.argv[0][:-15])+'bad_movies.npy')
    bad_movies = bad_movies.tolist()
    subset_size = 100
    Indecies = np.array([x for x in range(0, movie_max_id + 1)])
    permutation = np.loadtxt(movie_subset_path)
    #Todo: check legality of subset if not legal subset size should be 0
    permutation = permutation.reshape((-1)).astype(int)
    bad_indices = np.zeros((permutation.shape[0]), dtype=bool)
    for movieid,rating_num in bad_movies:
        indecies = permutation == movieid
        if np.sum(indecies) == 1:
            print_bad_movie(movieid=movieid, ratings_num=rating_num)
        bad_indices = np.logical_or(indecies, bad_indices)
    permutation = permutation[~(bad_indices.reshape(-1))]
    subset_size = permutation.shape[0]
    ajdacency_matrix = np.zeros((subset_size + 1, subset_size + 1))
    ajdacency_matrix = (correlations_indicator[permutation, :])[:, permutation]
    probabilty_matrix = np.zeros((subset_size + 1, subset_size + 1))
    probabilty_matrix = (correlations_data[permutation, :])[:, permutation]
    return ajdacency_matrix, probabilty_matrix, subset_size, \
           Indecies, correlations_data, movie_name

def cluster_print(movie_name,Indecies,clustering_vector,data_clusters):
    for c in data_clusters[1:]:
        cluster = (clustering_vector == c)
        cluster_names = ''
        for i in Indecies[cluster]:
            cluster_names = cluster_names + str(i) + ' '
            cluster_names = cluster_names + movie_name[i].decode('ascii') + ", "
        print(cluster_names[:-2])
    return

def pivot_clustering(ajdacency_matrix, probabilty_matrix, subset_size,Indecies,correlations_data,movie_name):
    W = np.ones((subset_size))
    W[0] = 0
    clustering_vector = np.zeros((movie_max_id + 1))
    current_cluster_number = 0

    while np.sum(W) != 0:
        v = np.random.choice(np.argwhere(W > 0).reshape(-1), size=1)[0]
        current_cluster_number = current_cluster_number + 1
        clustering_vector[ajdacency_matrix[0, v].astype(int)] = current_cluster_number
        movie_array = np.logical_and(ajdacency_matrix[v, :] == 1, W == 1)
        ajdacency_matrix[1:, movie_array] = 0
        ajdacency_matrix[movie_array, 1:] = 0
        W[movie_array] = 0
        clustering_vector[ajdacency_matrix[0, movie_array].astype(int)] = current_cluster_number

    data_clusters = np.unique(clustering_vector)
    sizes = []
    for x in data_clusters[1:]:
        cluster_size = np.sum((clustering_vector == x).astype(int))
        sizes.append((cluster_size))

    ############################
    cluster_print(movie_name,Indecies,clustering_vector,data_clusters)
    #############################3
    cost = 0
    for c in range(1, len(data_clusters) + 1):
        cluster = (clustering_vector == c)
        if np.sum(cluster.astype(int)) == 1:
            cost = cost + np.log(1 / (correlations_data[Indecies[cluster][0], Indecies[cluster][0]]))
        else:
            c_i = 0
            for t in Indecies[cluster]:
                for j in Indecies[cluster]:
                    if t != j:
                        if correlations_data[t, j] == 0:
                            correlations_data[t, j] = correlations_data[j, t]
                        c_i = c_i + np.log(1 / correlations_data[t, j])
            cost = cost + (c_i / 2) * (1 / (np.sum(cluster) - 1))
    print(cost)
    return

def build_linear_program(temp_ajd,temp_prob,lmbda,subset_size):
    ILP = pulp.LpProblem("clustering", pulp.LpMinimize)
    x = {}
    xi = {}
    for i in range(0, subset_size - 1):
        for j in range(i + 1, subset_size - 1):
            x[str(i) + ',' + str(j)] = pulp.LpVariable('x' + str(i) + str(j), cat='Continuous', lowBound=0.0,
                                                       upBound=1.0)
            xi[str(i) + ',' + str(j)] = pulp.LpVariable('xi' + str(i) + str(j), cat='Continuous', lowBound=0.0,
                                                        upBound=1.0)

    ILP += pulp.LpAffineExpression([(x[str(i) + ',' + str(j)], temp_ajd[i, j] * temp_prob[i, j])
                                    for i in range(0, subset_size - 1) for j in range(i + 1, subset_size - 1)])
    ILP += pulp.LpAffineExpression([(xi[str(i) + ',' + str(j)], lmbda * (1 - temp_ajd[i, j]) * (temp_prob[i, j]))
                                    for i in range(0, subset_size - 1) for j in range(i + 1, subset_size - 1)])


    for i in range(0, subset_size - 1):
        for j in range(i + 1, subset_size - 1):
            for w in range(0, subset_size - 1):
                if w != j and w != i:
                    ILP += x[str(i) + ',' + str(j)] <= x[str(min(i, w)) + ',' + str(max(i, w))] + x[
                        str(min(j, w)) + ',' + str(max(w, j))]
            ILP += xi[str(i) + ',' + str(j)] == 1 - x[str(i) + ',' + str(j)]

    return x, ILP

def calculate_cost_for_LP(clustering_vector,correlations_data,Indecies):
    costs = []
    for i in range(0, 10):
        cost = 0
        data_clusters = np.unique(clustering_vector[i, :])
        for c in range(1, len(data_clusters) + 1):
            cluster = (clustering_vector[i, :] == c)
            if np.sum(cluster.astype(int)) == 1:
                cost = cost + np.log(1 / (correlations_data[Indecies[cluster][0], Indecies[cluster][0]]))
            else:
                c_i = 0
                for t in Indecies[cluster]:
                    for j in Indecies[cluster]:
                        if t != j:
                            if correlations_data[t, j] == 0:
                                correlations_data[t, j] = correlations_data[j, t]
                            c_i = c_i + np.log(1 / correlations_data[t, j])
                cost = cost + (c_i / 2) * (1 / (np.sum(cluster) - 1))
        costs.append(cost)
    return costs


def LP_clustering(ajdacency_matrix, probabilty_matrix, subset_size,Indecies,correlations_data,movie_name):
    temp_ajd = ajdacency_matrix[1:, 1:]
    temp_prob = np.log(1 / (probabilty_matrix[1:, 1:] + 1e-10))
    lmbda = 10

    x, ILP = build_linear_program(temp_ajd,temp_prob,lmbda, subset_size)
    ILP.solve()

    clustering_vector = np.zeros((10, movie_max_id + 1))
    for i in range(0, 10):
        current_cluster_number = 0
        W = np.zeros(subset_size - 1)
        ajdacency_matrix_copy = np.copy(ajdacency_matrix)
        while (np.sum(1 - W) != 0):
            v = np.random.choice(np.argwhere(W == 0).reshape(-1), size=1)[0]
            current_cluster_number = current_cluster_number + 1
            W[v] = 1
            clustering_vector[i, ajdacency_matrix_copy[0, v + 1].astype(int)] = current_cluster_number
            for j in range(1, subset_size - 1):
                if v != j:
                    rand = np.random.rand(1)
                    if rand <= 1 - pulp.value(x[str(min(v, j)) + ',' + str(max(j, v))]):
                        clustering_vector[i, ajdacency_matrix_copy[0, j + 1].astype(int)] = current_cluster_number
                        W[j] = 1

    costs = calculate_cost_for_LP(clustering_vector,correlations_data,Indecies)
    np_costs = np.array(costs)
    cluster_print(movie_name, Indecies, clustering_vector[np.argmin(costs),:],
                  np.unique(clustering_vector[np.argmin(costs),:]))
    print(np.min(np_costs))
    return

def main():
    movie_subset_path = './'
    dataset_path = './'
    alg_type = 1
    if len(sys.argv) == 4:
        dataset_path = sys.argv[1]
        alg_type = int(sys.argv[2])
        movie_subset_path = sys.argv[3]
        ajdacency_matrix, probabilty_matrix, subset_size, Indecies, correlations_data, movie_name = load_and_clear_subset(
            movie_subset_path)
        if subset_size == 0:
            print("eror")
            #Todo:: throw error or exit
            exit(0)
        if alg_type == 1:
            pivot_clustering(ajdacency_matrix, probabilty_matrix, subset_size, Indecies, correlations_data, movie_name)
        elif alg_type == 2:
            LP_clustering(ajdacency_matrix, probabilty_matrix, subset_size, Indecies, correlations_data, movie_name)
        else:
            print("incorect algorithm input")


if __name__== '__main__':
    warnings.filterwarnings("ignore")
    main()
