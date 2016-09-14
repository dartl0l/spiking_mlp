# -*- coding: utf-8 -*-

import nest
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import multiprocessing

def norm_weights(weights):
    wt = np.array(weights)
    wt[wt < 0] = 0
    return weights / np.amax(np.sum(wt, axis=0))


def load_weights(rnd_state, n_layers, diehl_norm=False):
    weights = []
    for i in xrange(len(n_layers)):
        if diehl_norm is True:
            with open('rnd_state_' + str(rnd_state) +
                      '_layer_' + str(i) + '_w.txt', 'r') as weight:
                weights.append(norm_weights(np.loadtxt(weight)))
        else:
            with open('rnd_state_' + str(rnd_state) +
                      '_layer_' + str(i) + '_w.txt', 'r') as weight:
                weights.append(np.loadtxt(weight))
    return weights


def plot_input(X, y, plot2d=True, plot3d=False):
        pca_esti = decomposition.PCA(3)

        pcas = pca_esti.fit_transform(X)

        if plot3d is True:
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            plt.title('Classes')
            for pca, cl in zip(pcas, y):
                if cl == 1.0:
                    ax.scatter(pca[0], pca[1], pca[2], zdir='y',
                               depthshade=True, c='b', marker='^')
                elif cl == 0.0:
                    ax.scatter(pca[0], pca[1], pca[2], zdir='y',
                               depthshade=True, c='r', marker='^')
            ax.set_xlabel('First principal component')
            ax.set_ylabel('Second principal component')
            ax.set_zlabel('Third principal component')
            plt.show()
        if plot2d is True:
            fig = plt.figure(2)
            plt.title('Classes')
            for pca, cl in zip(pcas, y):
                if cl == 0.0:
                    plt.plot(pca[0], pca[1], 'r+')
                elif cl == 1.0:
                    plt.plot(pca[0], pca[1], 'bx')
            plt.xlabel('First principal component')
            plt.ylabel('Second principal component')
            plt.show()


def plot_output(X, result_list, neuron_classes, plot_real_classes=False):
    pca_esti = decomposition.PCA(3)

    pca = pca_esti.fit_transform(X)

    result, classes = get_class_from_network_output(result_list,
                                                    neuron_classes)

    plt.figure(1)
    plt.title('Our classes SNN')
    for pca, cl in zip(pca, classes):
        if cl == 0:
            plt.plot(pca[0], pca[1], 'r+', label='Class 1')
        elif cl == 1:
            plt.plot(pca[0], pca[1], 'bx', label='Class 2')
        else:
            plt.plot(pca[0], pca[1], 'k.', label='Unknown')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')

    if plot_real_classes is True:
        plt.figure(2)
        plt.title('Real classes')
        plt.plot(pca[0:50, 0], pca[0:50, 1], 'b.', label='Class 1')
        plt.plot(pca[50:100, 0], pca[50:100, 1], 'g.', label='Class 2')
        plt.plot(pca[100:150, 0], pca[100:150, 1], 'r.', label='Class 3')
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
    plt.show()


def get_class_from_network_output(output, neuron_classes, first_spike=True):
    classes = []
    result_list = []
    # print "========================================"
    for result in output:
        res_spikes = dict(zip(neuron_classes.keys(),
                              [[] for i in xrange(len(neuron_classes))]))
        for sender, time in zip(result[0]['senders'], result[0]['times']):
            res_spikes[sender].append(time)
        res_classes = dict.fromkeys(neuron_classes.values())
        for key in res_spikes:
            if first_spike:
                res_classes[neuron_classes[key]] = res_spikes[key]
            else:
                res_classes[neuron_classes[key]] = len(res_spikes[key])
        result_list.append(res_classes)
    if first_spike:
        for result in result_list:
            first_spike_time = 0
            cl = -1
            for i in xrange(len(neuron_classes)):
                if len(result[i]) != 0:
                    if first_spike_time == 0:
                        first_spike_time = result[i][0]
                        cl = i
                    elif result[i][0] < first_spike_time:
                        first_spike_time = result[i][0]
                        cl = i
            classes.append(cl)
    else:
        for result in result_list:
            k = 0
            for i in xrange(len(neuron_classes)):
                if result[i] == max(result):
                    k += 1
            if k > 1:
                classes.append(-1)
            else:
                classes.append(max(result, key=result.get))
    return result_list, classes


def show_result(result_list, Y, neuron_classes,
                show_spikes=True, first_spike=False):
    result, classes = get_class_from_network_output(result_list,
                                                    neuron_classes,
                                                    first_spike)

    if show_spikes is True:
        print "Results"
        for res, y in zip(result, Y):
            print "Class: {0}, ".format(y + 1),
            print "Spike count: ", res
            # for ress in res.keys():
            #     print "{0}: {1} |".format(res.keys()[0], res[res.keys()[0]]),
            # print

    err = 0.0
    for answer, y in zip(classes, Y):
        if answer != y:
            err += 1
        # print "Class: {0}, Answer {1}".format(y, answer)
#     print "Error: {0:.4f}".format(err / len(classes))
    error = err / len(classes)
    return error


# def search_threshold_freqs(X, y, sim_time, freq, thresh,
#                            neuron_classes, weights,
#                            n_layers,
#                            max_thresh=5.0, ht=0.5,
#                            max_freq=3000, hf=100,
#                            out_file_name='threshold_freqs_error.out',
#                            resolution=0.1, num_threads=4):
#     out_log = open(out_file_name, "w")
#     start_thresh = thresh
#     start_freq = freq
#     out_log.write("Frequency\tThreshold\tError\tStd\n")
#     while True:
#         out_log.write("=====================\n")
#         while True:
#             errors = sim_n(X, y, n_layers, weights,
#                            neuron_classes, sim_time, n_sims=3,
#                            th=thresh, max_freq=freq,
#                            resolution=resolution,
#                            num_threads=num_threads)
#             out_log.write(str(freq) + '\t' +
#                           str(thresh) + '\t' +
#                           str(np.mean(errors)) + '\t' +
#                           str(np.std(errors)) + '\n')
#             if thresh < max_thresh:
#                 thresh += ht
#             else:
#                 thresh = start_thresh
#                 break
#         if freq < max_freq:
#             freq += hf
#         else:
#             freq = start_freq
#             break
#     out_log.close()
#     print "search_finished"


def optimizing_threshold(X, y, sim_time, freq, thresh, n_layers, weights,
                         neuron_classes, out_file_name='stats.log',
                         n_threshs=5, n_freqs=10, h_freqs=100, h_threshs=1,
                         n_sim=5, save_ratio=True,
                         resolution=0.1, num_threads=4, first_spike=False):
    best_res = {'error': 1.0,
                'std': 0.0,
                'freq': 0.0,
                'thresh': 0.0}

    print "Begin"
    out_file = open(out_file_name, 'w')
    out_file.write("Freq\tThresh\tError\tStd\n")
    for j in xrange(0, n_freqs, 1):
        freq_cur = freq + j * h_freqs
        thresh_cur = thresh
        for i in xrange(1, n_threshs + 1, 1):
            if save_ratio is True:
                freq_curr = freq_cur * i
                thresh_curr = thresh * i
            else:
                freq_curr = freq_cur
                thresh_curr = thresh_cur + i * h_threshs
            errors = sim_n(X, y, n_layers, weights,
                           neuron_classes, sim_time,
                           n_sims=n_sim, th=thresh_curr,
                           max_freq=freq_curr, resolution=resolution,
                           num_threads=num_threads, first_spike=first_spike)
            out_file.write(str(freq_curr) + '\t' +
                           str(thresh_curr) + '\t' +
                           str(np.mean(errors)) + '\t' +
                           str(np.std(errors)) + '\n')
            if np.mean(errors) < best_res['error']:
                best_res['error'] = np.mean(errors)
                best_res['std'] = np.std(errors)
                best_res['freq'] = freq_curr
                best_res['thresh'] = thresh_curr
    out_file.close()
    print "End"
    return best_res


def spiking_cv(data, target, n_layers, neuron_classes, rnd_states,
               sim_time=1000.0, freq=1000.0, thresh=1.0,
               n_threshs=3, n_freqs=10, h_freqs=100, h_threshs=1, n_sim=5,
               diehl_norm=False, save_ratio=True, first_spike=False,
               resolution=0.1, num_threads=4,
               out_file_name='stats_cross.log'):
    # np.random.seed()
    # rng = np.random.randint(500)
    # rnd_states = range(rng, rng + n_rnd_states)
    print "Begin cross validation"
    out_file = open(out_file_name, 'w')
    out_file.write("Random state\tFreq\tThreshold\t" +
                   "Error Train\tStd Train\tError Test\tStd Test\n")
    for rnd_state in rnd_states:
        print rnd_state
        train_X, test_X, train_y, test_y = train_test_split(
            preprocessing.normalize(data),
            target, test_size=0.2,
            random_state=rnd_state,
            stratify=target)

        train_y = np.array(train_y)
        test_y = np.array(test_y)

        weights = load_weights(rnd_state, n_layers, diehl_norm=diehl_norm)

        best_res = optimizing_threshold(train_X, train_y, sim_time,
                                        freq, thresh, n_layers,
                                        weights, neuron_classes,
                                        'stats_' + str(rnd_state) + '.log',
                                        n_threshs, n_freqs, h_freqs, h_threshs,
                                        n_sim, save_ratio, resolution,
                                        num_threads, first_spike)
        print best_res

        errors = sim_n(test_X, test_y, n_layers, weights,
                       neuron_classes, sim_time, n_sim,
                       best_res['freq'], best_res['thresh'],
                       resolution=resolution, num_threads=num_threads,
                       first_spike=first_spike)

        out_file.write(str(rnd_state) + '\t' +
                       str(best_res['freq']) + '\t' +
                       str(best_res['thresh']) + '\t' +
                       str(best_res['error']) + '\t' +
                       str(best_res['std']) + '\t' +
                       str(np.mean(errors)) + '\t' +
                       str(np.std(errors)) + '\n')
    out_file.close()
    print "End cross validation"


def sim_n(X, y, n_layers, weights, neuron_classes,
          sim_time=10000.0, n_sims=3,
          max_freq=500.0, th=1000.0, plot=False,
          resolution=0.1, num_threads=4, first_spike=False):
    errors = []
    for i in xrange(n_sims):
        result_list = []
        workers_pool = multiprocessing.Pool(processes=num_threads)
        for i, freqs in enumerate(X):
            # print i
            workers_pool.apply_async(sim, (freqs, n_layers, weights, sim_time),
                                { # keyword arguments are passed to apply() in a dictionary
                                   "th": th, "max_freq": max_freq, "plot": plot,
                                   "resolution": resolution,
                                   "num_threads": 1}, 
                                callback=result_list.append)
        workers_pool.close() # prohibit sending new tasks after all were sent
        workers_pool.join() # wait for all subprocesses to stop
        error = show_result(result_list, y,
                            neuron_classes, False, first_spike=first_spike)
        errors.append(error)
    return errors


def sim(freqs, n_layers, weights, sim_time=1000.0, max_freq=500.0, th=1000.0,
        plot=False, resolution=0.1, num_threads=4):
    n_input = len(freqs)

    neuron_dict = {'model': 'iaf_psc_delta'}
    syn_dict = {'rule': 'all_to_all',
                'model': 'excitatory_static'}

    nest.ResetKernel()
    np.random.seed()
    rng = np.random.randint(500)
    nest.SetKernelStatus({'local_num_threads': num_threads,
                          # 'total_num_virtual_procs': 1,
                          'resolution': resolution,
                          'rng_seeds': range(rng, rng + num_threads)})

    poisson_generator_0 = nest.Create('poisson_generator', n_input)
    parrot_neuron_0 = nest.Create('parrot_neuron', n_input)

    layers = [nest.Create(neuron_dict['model'],
                          n_layer) for n_layer in n_layers]

    spike_detector_in = nest.Create('spike_detector')
    spike_detectors = [nest.Create('spike_detector') for n_layer in n_layers]

    voltmeters = [nest.Create('voltmeter', 1,
                              {'withgid': True, 'withtime': True})
                  for n_layer in n_layers]

    # layers_ids = [nest.GetStatus(layer, 'global_id') for layer in layers]
    # layers_ids = [nest.GetStatus(layer, 'local_id') for layer in layers]

    # print layers
    # print layers_ids

    nest.CopyModel('static_synapse', 'excitatory_static',
                   {'weight': 1.0, 'delay': 1.0})

    nest.Connect(poisson_generator_0, parrot_neuron_0,
                 'one_to_one', syn_spec='static_synapse')
    for voltmeter, layer in zip(voltmeters, layers):
        nest.Connect(voltmeter, layer)

    layer_a = parrot_neuron_0
    for layer in layers:
        layer_b = layer
        nest.Connect(layer_a, layer_b, syn_dict['rule'],
                     syn_spec=syn_dict['model'])
        layer_a = layer_b

    nest.Connect(parrot_neuron_0,
                 spike_detector_in, syn_dict['rule'])

    for layer, spike_detector in zip(layers, spike_detectors):
        nest.Connect(layer, spike_detector, syn_dict['rule'])

    nest.SetStatus(poisson_generator_0,
                   [{'rate': frequency * max_freq} for frequency in freqs])

    if neuron_dict['model'] == 'iaf_psc_exp':
        for layer in layers:
            nest.SetStatus(layer, {'V_m': 0.0, 'E_L': 0.0,  'I_e': 1.0,
                                   'tau_m': 100.0, 't_ref': 3.0, 'C_m': 1.0,
                                   'V_reset': 0.0, 'V_th': th})

    elif neuron_dict['model'] == 'iaf_psc_delta':
        for layer in layers:
            nest.SetStatus(layer, {'V_m': 0.0, 'E_L': 0.0,  'I_e': 0.0,
                                   'tau_m': sim_time, 't_ref': 0.03,
                                   'C_m': 0.01, 'V_reset': 0.0, 'V_th': th})
        # nest.SetStatus(layers[-1], {'V_th': 100.0})

    elif neuron_dict['model'] == 'iaf_psc_alpha':
        for layer in layers:
            nest.SetStatus(layer, {'V_m': 0.0, 'E_L': 0.0,  'I_e': 1.0,
                                   'tau_m': 60.0, 't_ref': 3.0, 'C_m': 1.0,
                                   'V_reset': 0.0, 'V_th': th})
    # else:
    #     for layer in layers:
    #         nest.SetStatus(layer, {'V_m': 0.0, 'E_L': 0.0,  'I_e': 0.0,
    #                                'tau_m': sim_time, 't_ref': 0.0,
    #                                'C_m': 1.0, 'V_reset': 0.0, 'V_th': th})

    layers_a = [parrot_neuron_0] + layers[:-1]
    for layer, layer_weights, layer_ids in zip(layers_a, weights, layers):
        for neuron, weights in zip(layer, layer_weights):
            for layer_id, weight in zip(layer_ids, weights):
                nest.SetStatus(nest.GetConnections([neuron],
                               target=[layer_id]), 'weight', weight)

    # for layer, layer_weights, layer_ids in zip(layers_a, weights, layers_ids):
    #     for neuron, weights in zip(layer, layer_weights):
    #         for layer_id, weight in zip(layer_ids, weights):
    #             nest.SetStatus(nest.GetConnections([neuron],
    #                            target=[layer_id]), 'weight', weight)

    nest.Simulate(sim_time)
    if plot:
        print "input spikes"
        nest.raster_plot.from_device(spike_detector_in)
        nest.raster_plot.show()
        # for i, layer in enumerate(n_layers):
        for i, voltmeter in enumerate(voltmeters[:-1]):
            print "hidden layer " + str(i) + " membrane potential"
            nest.voltage_trace.from_device([voltmeter[0]])
            nest.voltage_trace.show()
        print "output layer membrane potential"
        for voltmeter in voltmeters[-1]:
            nest.voltage_trace.from_device([voltmeter])
            nest.voltage_trace.show()
        # print "spikes out"
        # nest.raster_plot.from_device(spike_detectors[-1], hist=True)
        # nest.raster_plot.show()
    return nest.GetStatus(spike_detectors[-1], keys='events')
