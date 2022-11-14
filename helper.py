import csv
import numpy as np


def print_results(name, techniques, metrics, label_names):
    with open(name+'.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for metric in metrics.keys():
            print(metric)
            temp_metric = np.array(metrics[metric])
            for i in range(len(techniques)):
                label_score = []
                for label in range(len(label_names)):
                    tempo = [k for k in temp_metric[:, i, label]
                             if str(k) != str(np.average([]))]
                    if len(tempo) == 0:
                        tempo.append(0)
                    label_score.append(np.array(tempo))
                temp_mean = []
                for k in label_score:
                    temp_mean.append(k.mean())

                temp_mean = np.array(temp_mean).mean()
                writer.writerow([techniques[i], metric, temp_mean] +
                                [label_score[o].mean() for o in range(len(label_names))])
                print(techniques[i], ' {} | {}'.format(round(temp_mean, 5), ' '.join(
                    [str(round(label_score[o].mean(), 5)) for o in range(len(label_names))])))


def print_results_ap(ma, label_names, conf):
    num_instances = len(ma['FTP'])
    labelset_size = len(label_names)

    confs_mean = []
    temp_metric = np.array(ma['FTP'])
    for i in range(len(conf)):
        label_score = []
        for label in range(labelset_size):
            tempo = [k for k in temp_metric[:, i, label]
                     if str(k) != str(np.average([]))]
            if len(tempo) == 0:
                tempo.append(0)
            label_score.append(np.array(tempo))
        temp_mean = []
        for k in label_score:
            temp_mean.append(k.mean())
        confs_mean.append(np.array(temp_mean).mean())

    confs_mean2 = []
    temp_metric = np.array(ma['NZW'])
    for i in range(len(conf)):
        label_score = []
        for label in range(labelset_size):
            tempo = [k for k in temp_metric[:, i, label]
                     if str(k) != str(np.average([]))]
            if len(tempo) == 0:
                tempo.append(0)
            label_score.append(np.array(tempo))
        temp_mean = []
        for k in label_score:
            temp_mean.append(k.mean())
        confs_mean2.append(np.array(temp_mean).mean())
    maxx = np.argmax(confs_mean)

    if 'AUPRC' in ma:
        confs_mean3 = []
        temp_metric = np.array(ma['AUPRC'])
        for i in range(len(conf)):
            label_score = []
            for label in range(labelset_size):
                tempo = [k for k in temp_metric[:, i, label]
                         if str(k) != str(np.average([]))]
                if len(tempo) == 0:
                    tempo.append(0)
                label_score.append(np.array(tempo))
            temp_mean = []
            for k in label_score:
                temp_mean.append(k.mean())
            confs_mean3.append(np.array(temp_mean).mean())
        print('Baseline:', confs_mean[0], ' and NZW:',
              confs_mean2[0], 'and AUPRC:', confs_mean3[0])
        print('Max Across:', confs_mean[maxx], ' and NZW:',
              confs_mean2[maxx], 'and AUPRC:', confs_mean3[maxx])
    else:
        print('Baseline:', confs_mean[0], ' and NZW:', confs_mean2[0])
        print('Max Across:', confs_mean[maxx], ' and NZW:', confs_mean2[maxx])

    # per label per instance no softmax
    ftp = []
    nzw = []
    auprc = []
    for i in range(num_instances):
        maxx = np.argmax(ma['FTP'][i], axis=0)
        ftp.append(np.max(ma['FTP'][i], axis=0))
        nzw.append([np.array(ma['NZW'][i])[:, j][maxx[j]]
                   for j in range(labelset_size)])
        if 'AUPRC' in ma:
            auprc.append([np.array(ma['AUPRC'][i])[:, j][maxx[j]]
                         for j in range(labelset_size)])
    ftp = np.array(ftp)
    nzw = np.array(nzw)
    auprc = np.array(auprc)
    av_ftp = 0
    av_nzw = 0
    av_auprc = 0
    for label in range(labelset_size):
        av_ftp += np.mean([k for k in ftp[:, label] if str(k) != 'nan'])
        av_nzw += np.mean([k for k in nzw[:, label] if str(k) != 'nan'])
        if 'AUPRC' in ma:
            bob = np.mean([k for k in auprc[:, label] if str(k) != 'nan'])
            if str(bob) != 'nan':
                av_auprc += bob
    if 'AUPRC' in ma:
        print('Per Label Per Instance:', av_ftp/labelset_size, ' and NZW: ',
              av_nzw/labelset_size, 'and AUPRC:', av_auprc/labelset_size)
    else:
        print('Per Label Per Instance:', av_ftp/labelset_size,
              ' and NZW: ', av_nzw/labelset_size)

    # per instance not per label no softmax
    avg_ftp = []
    avg_nzw = []
    avg_auprc = []
    for i in range(num_instances):
        instance_level = []
        instance_level2 = []
        instance_level3 = []
        for j in range(len(ma['FTP'][i])):
            test = [k for k in ma['FTP'][i][j] if str(k) != 'nan']
            test2 = [k for k in ma['NZW'][i][j] if str(k) != 'nan']
            if 'AUPRC' in ma:
                test3 = [k for k in ma['AUPRC'][i][j] if str(k) != 'nan']
            test = np.mean(test)
            test2 = np.mean(test2)
            instance_level.append(test)
            instance_level2.append(test2)
            if 'AUPRC' in ma:
                instance_level3.append(np.mean(test3))
        maxx = np.argmax(instance_level)

        avg_ftp.append(ma['FTP'][i][maxx])
        avg_nzw.append(ma['NZW'][i][maxx])
        if 'AUPRC' in ma:
            avg_auprc.append(ma['AUPRC'][i][maxx])
    avg_ftp = np.array(avg_ftp)
    avg_nzw = np.array(avg_nzw)
    avg_auprc = np.array(avg_auprc)
    av_ftp = 0
    av_nzw = 0
    av_auprc = 0
    for label in range(labelset_size):
        av_ftp += np.mean([k for k in avg_ftp[:, label] if str(k) != 'nan'])
        av_nzw += np.mean([k for k in avg_nzw[:, label] if str(k) != 'nan'])
        if 'AUPRC' in ma:
            bob = np.mean([k for k in avg_auprc[:, label] if str(k) != 'nan'])
            if str(bob) != 'nan':
                av_auprc += bob
    if 'AUPRC' in ma:
        print('Per Instance:', av_ftp/labelset_size, ' and NZW: ',
              av_nzw/labelset_size, 'and AUPRC:', av_auprc/labelset_size)
    else:
        print('Per Instance:', av_ftp/labelset_size,
              ' and NZW: ', av_nzw/labelset_size)
