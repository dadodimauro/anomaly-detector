import importlib

# https://github.com/KulikDM/pythresh
def get_labels(th_algorithm, y_pred, max_contamination=0.1):
    th_algorithms = []
    if type(th_algorithm) is list:  # COMB
        for element in th_algorithm:
            module = importlib.import_module('.'+th_algorithm, package='pythresh.thresholds')
            th_algorithm = getattr(module, th_algorithm.upper())
            th_algorithms.append(th_algorithm)
            
        module = importlib.import_module('.comb', package='pythresh.thresholds')
        th_algorithm = getattr(module, 'COMB')
        
        thres = th_algorithm(th_algorithms)
    elif th_algorithm == 'all':  # all algorithms            
        module = importlib.import_module('.'+th_algorithm, package='pythresh.thresholds')
        th_algorithm = getattr(module, th_algorithm.upper())
        
        thres = th_algorithm(max_contam=max_contamination)
        
    else:  # single algorithm
        module = importlib.import_module('.'+th_algorithm, package='pythresh.thresholds')
        th_algorithm = getattr(module, th_algorithm.upper())

        thres = th_algorithm()
    
    labels = thres.eval(y_pred)

    return labels