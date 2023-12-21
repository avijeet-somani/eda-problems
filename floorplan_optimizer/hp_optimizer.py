import train
from train import create_datasets, batch_test, batch_train, get_model
import optuna
from optuna.trial import TrialState
import optuna.visualization as vis
import matplotlib.pyplot as plt
import torch
import utils



class HPOptimizer :
    def __init__(self, args) : 
        self.args = args
        self.train_dataset = None
        self.test_dataset = None

    def kickstart(self):
        self.optimize_params()

    def optimize_params(self) : 

        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = "distributed_TSP"
        storage_name = f"sqlite:///{study_name}.db"
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, sampler=sampler, load_if_exists=True)
        
        self.train_dataset, self.test_dataset = create_datasets(self.args)

        
        # Start the optimization process with parallel execution.
        n_jobs = self.args.hp_parallelism # Set the number of parallel jobs as needed.
        study.optimize(self.objective, n_trials=self.args.hp_trials, n_jobs=n_jobs)
        #study.optimize(self.objective, n_trials=3, timeout=600)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])    
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]) 
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        #self.visualize(study)


    def visualize(self, study):
        #visualization     
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        #fig3 = optuna.visualization.plot_pareto_front(study)
        fig1.savefig('optimization_history_plot.png')
        fig2.savefig('optimization_param_importance.png')
        #fig3.savefig('optimization_pareto_front.png')
        plt.show()





    def define_model(self, trial) : 
        #model hyperparams : embedding_size, hidden_size
        #embedding_size = trial.suggest_categorical("embedding_size", [32, 64,128]) 
        #hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        embedding_size = trial.suggest_categorical("embedding_size", [ 4, 8, 16]) 
        hidden_size = trial.suggest_categorical("hidden_size", [4, 8, 16])

        model = get_model(self.args.model_type, embedding_size, hidden_size)
        
        return model
    

    def define_optimizer(self, trial) :
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        #optimizer_name = "SGD"
        learning_rate =  trial.suggest_categorical("learning_rate", [3.0*1e-4, 1e-5, 1e-1])
        return  [ optimizer_name, learning_rate ]
        #optimizer_name = torch.optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    def objective(self, trial) : 
        model = self.define_model(trial)
        optimizer_name, learning_rate = self.define_optimizer(trial)
        optimizer = getattr(torch.optim, optimizer_name) (model.parameters(), lr=learning_rate)
        reward = batch_test(self.test_dataset, model)
        print("AVG Tour Distance before Training", reward)

        batch_train(self.args,  self.train_dataset, model , optimizer )
        reward = batch_test( self.test_dataset, model) #reward is the tour distance
       
        print("AVG Tour Distance after Training", reward)

        model_signature = f'model_trial_{trial.number}.h5'
        utils.save_model(model, model_signature)
        return reward


