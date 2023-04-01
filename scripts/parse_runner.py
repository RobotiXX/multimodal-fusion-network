import pickle

with open('/Users/bhabaranjanpanigrahi/Research/Code/fusion-network/recorded-data/133231.bag/snapshot.pickle', 'rb') as f:
    x = pickle.load(f)
    # print(len(x[1]['point_cloud'][1]))
    # print(len(x[1]['point_cloud'][0]))
    print(x[1].keys())