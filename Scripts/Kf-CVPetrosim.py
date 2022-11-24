import roxar, roxar.jobs
import random

# pdb.set_trace()
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from itertools import groupby
from operator import itemgetter

#project_path = r"C:/Users/sicacha/Documents/Emerald13_1copy.pro"
#project = roxar.Project.open_import(project_path)

dummy = '********************Generate CV automatically******************************'


class GenerateCrossValidate:
    def __init__(self, grid_model, blocked_well_name, seed, nfolds, orig_jobs_names, grid_name,real,modes):
        self.grid_model = grid_model
        self.blocked_well_name = blocked_well_name
        self.blocked_wells = grid_model.blocked_wells_set[blocked_well_name]
        self.seed = seed
        self.nfolds = nfolds
        self.orig_jobs_names = orig_jobs_names
        self.grid_name = grid_name
        self.real = real
        self.modes = modes

    def PredGen(self):
        for names in self.orig_jobs_names:
            cont = 1
            orig_job = roxar.jobs.Job.get_job(owner=['Grid models', self.grid_name, 'Grid'],
                                              type='Petrophysical Modeling',
                                              name=names)

            for m in self.modes:
                #print(m)
                for fold in self.folds:
                    params = orig_job.get_arguments(False)
                    self.props = params['VariableNames']
                    params['UseAllWells'] = False
                    params['Algorithm'] = m
                    params['PrefixOutputName'] = m + '_' + str(self.nfolds) + '-Fold_' + params['PrefixOutputName'] + '_CV_' + str(cont)
                    params['WellSelection'] = [str(y) for y in np.array(set(self.all_well_names) - set(fold)).tolist()]
                    # Change parameters for the new job
                    new_job = roxar.jobs.Job.create(owner=['Grid models', self.grid_name, 'Grid'],
                                                    type='Petrophysical Modeling',
                                                    name= names + '_CV_' + str(cont))
                    new_job.set_arguments(params)

                    new_job.save()
                    if m == 'SIMULATION':
                        for r in self.real:
                            new_job.execute(realisation=r)
                    else:
                        new_job.execute(realisation=0)
                    cont = cont + 1

    def FoldGen(self):
        self.all_well_names = np.array(self.blocked_wells.get_well_names())
        random.shuffle(self.all_well_names)
        k = int(np.ceil(len(self.all_well_names) / self.nfolds))
        ss = list(range(1, self.nfolds + 1))
        foldvec = np.tile(ss, k)[0:len(self.all_well_names)]
        self.folds = [self.all_well_names[np.where(foldvec == s)[0]] for s in ss]
        #print(self.folds)


class AnalyzeCrossValidate:
    def __init__(self, grid_model, blocked_well_name, folds, orig_job_names, grid_name,real,modes,props):
        self.grid_model = grid_model
        self.folds = folds
        self.nfolds = len(self.folds)
        self.blocked_well_name = blocked_well_name
        self.orig_jobs_names = orig_job_names
        self.dict_prop_and_rem_well_names = {}
        self.dict_prop_and_rem_well_comp = {}
        #self.fold_comp_true_vals = {}
        self.fold_comp_vals = {}
        self.fold_comp_summ = {}
        self.grid_name = grid_name
        self.blocked_wells = grid_model.blocked_wells_set[blocked_well_name]
        self.all_well_names = np.array(self.blocked_wells.get_well_names())
        self.modes = modes
        self.props = props
        if self.modes == 'PREDICTION':
            self.real = range(0,1)
        else:
            self.real = real


    def MakeCompVals(self):
        for m in self.props:
            for names in self.orig_jobs_names:
                for l in range(1, self.nfolds + 1):
                    job_tmp = roxar.jobs.Job.get_job(owner=['Grid models', 'Heterogeneity', 'Grid'],
                                                     type='Petrophysical Modeling',
                                                     name=names + '_CV_' + str(l))
                    params = job_tmp.get_arguments(False)
                    rem_wells = [str(y) for y in np.array(set(self.all_well_names) - set(params['WellSelection'])).tolist()]
                    all_bw_cell_numbers = self.blocked_wells.get_cell_numbers()
                    filter = self.blocked_wells.get_data_indices(rem_wells)
                    filter_cell_nums = all_bw_cell_numbers[filter]
                    for r in self.real:
                        #print(params['PrefixOutputName'] + '_' + m)
                        property = self.grid_model.properties[params['PrefixOutputName'] + '_' + m]
                        prop_vals = property.get_values(realisation=r, cell_numbers=filter_cell_nums)  ##Here
                        wellp = self.blocked_wells.properties[m]
                        well_values = wellp.get_values()
                        well_values2 = well_values[filter]
                        well_final = np.zeros_like(prop_vals)  # Function getwellvalues in oldcode
                        i = 0
                        while i < len(well_final):
                            if well_values2[i] is np.ma.masked:
                                well_final[i] = prop_vals[i]
                            else:
                                well_final[i] = well_values2[i]
                            i += 1
                        self.fold_comp_vals[tuple(rem_wells),m, names, 'Fold ' + str(l), r] = [prop_vals, well_final]
                        #self.fold_comp_pred_vals[tuple(rem_wells),m,names, 'Fold ' + str(l), r] = [prop_vals]
                        #self.fold_comp_true_vals[tuple(rem_wells),m,names, 'Fold ' + str(l), r] = [well_final]
                        if self.modes == 'PREDICTION':
                            self.fold_comp_summ[tuple(rem_wells),m, names, 'Fold ' + str(l), r] = [
                                np.mean(abs(prop_vals - well_final)),
                                np.sqrt(np.mean((prop_vals - well_final) ** 2))]
        #print(self.fold_comp_summ)
        #print(self.fold_comp_vals)

    def Run(self):
        self.MakeCompVals()

    def PrintInfo(self):
        # Example
        for key in self.fold_comp_summ:
            print(key[1])
            print(f'Removed wells: {key[0]}')
            print(f'Original job name: {key[2]}')
            print(f'Original RMS Parameter name: {key[3]}')
            print(f'Fold MAE = {self.fold_comp_summ[key][0]}')
            print(f'Fold RMSE = {self.fold_comp_summ[key][1]}')
            print('*****************************************************************')




    def Analyze(self):

        def CRPS():
            values = list(value)
            print('CRPS')
            #print(values)
            #print(len(values))
            #print([len(x) for x in values[0]])
            crps_list = [self.fold_comp_vals[values[pos]][0] for pos in self.real]
            #print(len(crps_list))
            #print(len(crps_list[0]))
            true_list = [self.fold_comp_vals[values[pos]][1] for pos in self.real]
            #print(len(true_list))
            #print(len(true_list[0]))
            crps_final_list = list()
            for i in range(0, len(crps_list[0])):
                df = [crps_list[p][i] for p in self.real]
                ecdf = ECDF(df)
                x = ecdf.x[1:]
                y = ecdf.y[1:]
                xseq = np.linspace(np.min(x), np.max(x), num=1000)
                xsize = xseq[1] - xseq[0]
                idx = np.array([xx <= true_list[0][i] for xx in xseq])
                yseq = np.array([y[np.max(np.where(x <= xx))] for xx in xseq])
                yseq[np.where(~idx)[0]] = 1 - yseq[np.where(~idx)[0]]
                crps = sum(yseq * xsize)
                crps_final_list.append(crps)
            return np.mean(crps_final_list)

        def MAE(ll):
            return np.mean(abs(ll[0] - ll[1]))

        def RMSE(ll):
            return np.sqrt(np.mean((ll[0] - ll[1]) ** 2))
        #print('Here will the analysis come')
        #print('NBNB well data is stored in masked numpy array, but property is stored normal numpy array.')
        #print('NBNB cont See GetWellValues function how to fix this')

        group_job_var = groupby(self.fold_comp_vals,
                                key=itemgetter(1,2,3))
        self.dict_job_comp = {}
        for key, value in group_job_var:
            if self.modes == 'PREDICTION':
                ll = self.fold_comp_vals[list(value)[0]]
                self.dict_job_comp[key]=[MAE(ll),RMSE(ll)]
            else:
                self.dict_job_comp[key] = [CRPS()]
        #print(self.dict_job_comp)


    def ModChoice(self):
        self.group_fold_var = groupby(self.dict_job_comp, key=itemgetter(0,1))
        self.final_comp = {}
        self.job_prop_summ = {}

        for key, value in self.group_fold_var:
            values = list(value)
            val_list = [self.dict_job_comp[values[pos]] for pos in range(0, self.nfolds)]
            #print(val_list)
            #self.job_prop_summ[key] = np.mean(val_list)
            self.job_prop_summ[key] =[np.mean([x[i] for x in val_list]) for i in range(0,len(val_list[0]))]

        self.group_prop = groupby(self.job_prop_summ,
                             key=itemgetter(0))

        for key, value in self.group_prop:
            values = list(value)
            val_list = [self.job_prop_summ[values[pos]] for pos in range(0, len(self.orig_jobs_names))]
            #print(val_list)
            #[np.mean([x[i] for x in val_list]) for i in range(0, len(val_list[0]))]
            #best_job = self.orig_jobs_names[np.argmin(val_list)]
            best_crit = [np.min([x[i] for x in val_list]) for i in range(0, len(val_list[0]))]
            best_job = [self.orig_jobs_names[y] for y in
                        [np.argmin([x[i] for x in val_list]) for i in range(0, len(val_list[0]))]]
            #print(best_job)
            if self.modes == 'SIMULATION':
                print(f'Best job for parameter {key} in CRPS is {best_job}; CRPS = {np.min(val_list)}')
            else:
                print(f'Best job for parameter {key} in MAE is {best_job[0]}; MAE = {best_crit[0]}')
                print(f'Best job for parameter {key} in RMSE is {best_job[1]}; RMSE = {best_crit[1]}')

    def CleanUpRMSProject(self):
        for m in self.modes:
            for names in self.orig_jobs_names:
                for f in range(1,self.nfolds+1):
                    job_tmp = roxar.jobs.Job.get_job(owner=['Grid models', 'Heterogeneity', 'Grid'],
                                                     type='Petrophysical Modeling',
                                                     name=names + '_CV_' + str(f))
                    params = job_tmp.get_arguments(False)
                    props = params['VariableNames']
                    print(params["PrefixOutputName"])
                    for p in props:
                        del self.grid_model.properties[params["PrefixOutputName"]]


dummy = '********************Petrosim Advanced Cross Validation input parameters start **********************************'


class InputModelACV:
    def __init__(self):
        self.grid_name = 'Heterogeneity'
        self.blocked_well_name = 'BW'
        self.debug_info = False
        self.seed = 1036610602
        self.nfolds = 2
        self.orig_job_names = ['Original4', 'Original5', 'Original6']
        self.real = list(range(0,5))
        self.modes = ["PREDICTION"]
        #self.save_realizations_path = 'C:/Users/sicacha/Documents/Data/Data'

    def IsOk(self):
        return True  # NBNB fjellvoll add later


dummy = '********************Petrosim Advanced Cross Validation input parameters end **********************************'


def Main():
    im = InputModelACV()
    print(im.IsOk())
    print('********************Petrosim Advanced Cross Validation running **********************************')
    grid_model = project.grid_models[im.grid_name]

    if im.IsOk():
        gcv = GenerateCrossValidate(grid_model, im.blocked_well_name, im.seed, im.nfolds, im.orig_job_names,
                                    im.grid_name,im.real,im.modes)
        gcv.FoldGen()
        gcv.PredGen()
        for m in im.modes:
            print(m)
            acv = AnalyzeCrossValidate(grid_model, im.blocked_well_name, gcv.folds, im.orig_job_names, im.grid_name,im.real,
                                       m,gcv.props)
            acv.Run()
            acv.Analyze()
            if m=='PREDICTION':
                acv.PrintInfo()
            acv.ModChoice()
            #acv.CleanUpRMSProject()

    print('********************end Petrosim Advanced Cross Validation **************************************')


Main()



