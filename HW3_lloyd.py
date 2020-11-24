import matplotlib.pyplot as plt
import numpy as np

from mnist import MNIST

# def load_dataset():
mndata = MNIST('./')
mndata.gz = True

X_train_raw, labels_train_raw = mndata.load_training()
X_test_raw, labels_test_raw = mndata.load_testing()
X_train_raw = np.array(X_train_raw)/255.0
X_test_raw = np.array(X_test_raw)/255.0

def find_closest(x, centers):
    #find which center is closest to each point
    distance = np.sqrt(((x - centers[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distance, axis=0)

def move_centers(x, closest, centers, k):
    #move the centers
    new_centers = np.zeros(centers.shape)
    for i in range(k):
        new_centers[i] = np.mean(x[closest==i], axis=0)
    return new_centers

def lloyd(x, centers, k):
    #run lloyds algorithm
    error_list = []
    counter = 0
    while True:
        closest = find_closest(x, centers)
        new_centers = move_centers(x, closest, centers, k)
        error = find_error(x, new_centers,k)
        error_list.append(error)
        if counter % 100 == 0:
            print(error)
        counter += 1
        #done when we haven't updated any steps
        if(np.array_equal(new_centers, centers)):
            return new_centers, error_list
        centers = new_centers
        
def find_error(x, centers,k):
    #calculate there based on equatoin (1) from HW
    error = 0
    closest = find_closest(x, centers)
    for i in range(k):
        points = x[np.where(closest == i)]
        error += np.linalg.norm(points - centers[i])**2
    return error
        
    
x = X_train_raw
k_list = [32, 64]#[2, 4, 8, 16, 32, 64]
train_error_list = []
test_error_list = []
for k in k_list:
    print("k = ", k)
    #pick centers at random
    idx = np.random.choice(len(x), k)
    centers = x[idx,:]

    #run algorithm
    centers, error_list = lloyd(x,centers, k)
    
    
    train_error = find_error(x, centers,k)
    test_error = find_error(X_test_raw, centers,k)
#     print("train_error = ", train_error)
#     print("test_error = ", test_error)
    
    train_error_list.append(train_error)
    test_error_list.append(test_error)
    
print("train_error_list = ", train_error_list)
print("test_error_list = ", test_error_list)