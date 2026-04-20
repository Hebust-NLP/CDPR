class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)
        
    def _get_hyper_parameters(self, args):
        """
        Args:
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
            num_train_epochs (int): The number of training epochs.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            aligned_method (str): The method for aligning different modalities. ('ctc', 'conv1d', 'avg_pool')
            weight_decay (float): The coefficient for L2 regularization. 
        """
        hyper_parameters = {
            # common parameter
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'eval_monitor': ['acc'], 
            'train_batch_size': [16],
            'eval_batch_size': [8],
            'test_batch_size': [8],
            'wait_patience': [5],
            'num_train_epochs': [40],
            'num_modalities': [3],
            'need_aligned': [True],
            'aligned_method': ['ctc'],
            # train parameters
            'warmup_proportion': [0.01],
            'lr': [7e-6],
            'weight_decay': [0.1],
            'weight_hidden_dim': [768],
            'weight_dropout': [0.2],  
            'temperature': [5.0],    
            'diff_weight': [0.01],
            'sim_weight': [0.01],
            'uni_weight': [0.1],
            'reasoning_weight': [0.1],
        } 

        return hyper_parameters 
     
