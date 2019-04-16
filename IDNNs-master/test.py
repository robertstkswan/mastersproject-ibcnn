import idnns.plots.plot_figures as plt_fig
import time

if __name__ == '__main__':
    start = time.time()

    str_name = [[
                    'jobs/net_sampleLen=1_nDistSmpls=1_layerSizes=10,7,5,4,3_nEpoch=8000_batch=512_nRepeats=1_nEpochInds=274_LastEpochsInds=7999_DataName=var_u_lr=0.0004/']]
    plt_fig.plot_alphas(str_name[0][0])

    # plt_fig.plot_figures(str_name, 2, 'd')
    # mode = 2
    # save_name = 'figure'
    # plt_fig.plot_figures(str_name, mode, save_name)
    # plt_fig.plot_hist(str_name[0][0])

    end = time.time()
    print("Time taken for main.py is " + str(end - start))
