
# Imports...
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, \
                                 HuberRegressor, SGDRegressor, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
                             AdaBoostRegressor, RandomForestRegressor, \
                             GradientBoostingRegressor, StackingClassifier
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, \
                            accuracy_score, precision_score, recall_score
# sklearn tools for the care and feeding of models...
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import scale
# Scipy stuff we're going to need
from scipy.stats import linregress
from scipy import interpolate
# Constants:
my_random_seed = 112358

# A function to do the demographic correction.
def npx_dc_correction( npx, non_npx, holdout_non_npx_columns = [ "endpoint" ], train_col_name = "endpoint", log_transform = True, train_on_neg_samp = True, save_file_name = None, outlier_pct = None, verbose = True ):
    """
    Perform the demographic and clinical (DC) corrections to NPX data.
    """
    # Make it clear how many biomarkers we're correcting.
    if verbose:
        print( "Performing a demographic/clinical correction for", npx.shape[1], "biomarkers..." )
    # We're only going to DC correct NPX values using features in `non_npx`.
    dc_corr_feats = [ column for column in non_npx.columns if ( ( column not in holdout_non_npx_columns ) and ( len( non_npx[ column ].unique() ) >= 2 ) ) ]
    if ( len( dc_corr_feats ) == 0 ):
        print( "Features in dataset have no variations. DC correction is not possible." )
        return npx
    # If we have an outlier definition, flag the outiers. Excluding dataframes with exclusively binary features from the outlier condition.
    if outlier_pct is not None and not ( all( x <= 2 for x in list( non_npx[ dc_corr_feats ].nunique() ) ) ):
        not_outliers = np.concatenate( [ ( ( non_npx[ feat ] > np.percentile( non_npx[ feat ][ non_npx[ feat ].notnull() ], outlier_pct ) ) & \
                                           ( non_npx[ feat ] < np.percentile( non_npx[ feat ][ non_npx[ feat ].notnull() ], 100. - outlier_pct ) ) ).values.reshape( (-1, 1) ) \
                                         for feat in dc_corr_feats if len(non_npx[ feat ].unique()) > 2 ], axis = 1 ).all( axis = 1 )
    else:
        not_outliers = np.ones( non_npx.shape[0] ).astype( np.bool )
    # Option to train only on the negative sameples ( endpoint = 0 )
    if train_on_neg_samp:
        low_da_rows = ( non_npx[ train_col_name ] == 0 ).values
    else:
        low_da_rows = np.ones_like( non_npx[ train_col_name ].values ).astype( np.bool )

    # Initialize a dataframe to store the corrected NPX values.
    corrected_npx = npx.copy()
    for column in corrected_npx.columns:
        corrected_npx[ column ] = [ np.nan ] * corrected_npx.shape[ 0 ]

    # No readout of statistics
    if verbose:
        if outlier_pct is not None:
            print( "Extracting residuals via OLS regression adjustment by", dc_corr_feats, "for central {:2.1f}% of data...".format( 100. - ( 2. * outlier_pct ) ) )

    # If we've got an output file name, get ready to readout demographic adjustment statistics.
    if save_file_name is not None:
        # Initialize dataframe to export to CSV
        demoadj_df = pd.DataFrame()

    # Initialize a dictionary to store the DC correction models so we can hand them back to the 
    # main body of the code.
    dc_corr_models = { bmkr: LinearRegression( fit_intercept = True, n_jobs = -1 ) for bmkr in npx.columns }
    # Iterate over the biomarkers in npx.
    for bmkr in npx.columns:
        # Flag the nans in each row so that we don't break the model training.
        not_nans = non_npx[ dc_corr_feats ].notnull().any( axis = 1 ).values & npx[ bmkr ].notnull().values
        # Train the `dc_corr_models[ bmkr ]` on the low-desease-activity rows.
        if verbose:
            print( "    Non-nans:", np.count_nonzero( not_nans ) )
            print( "Non-outliers:", np.count_nonzero( not_outliers ) )
            print( "      Low-DA:", np.count_nonzero( low_da_rows ) )
            print( "Full Overlap:", np.count_nonzero( not_nans & not_outliers & low_da_rows ) )
        # Fit the model. Make sure to do the log transform if we have been called to do so.
        if log_transform:
            dc_corr_models[ bmkr ].fit( non_npx[ dc_corr_feats ][ not_nans & not_outliers & low_da_rows ], np.log10( npx[ bmkr ][ not_nans & not_outliers & low_da_rows ] ) )
            npx_preds = pd.Series( data = dc_corr_models[ bmkr ].predict( non_npx[ dc_corr_feats ] ), index = npx.index, name = bmkr )
            corrected_npx[ bmkr ] = np.log10( npx[ bmkr ] ) - npx_preds
        else:
            dc_corr_models[ bmkr ].fit( non_npx[ dc_corr_feats ][ not_nans & not_outliers & low_da_rows ], npx[ bmkr ][ not_nans & not_outliers & low_da_rows ] )
            # Use `dc_corr_models[ bmkr ]` and all the _non-nan_ feature rows to predict those NPX values.
            npx_preds = pd.Series( data = dc_corr_models[ bmkr ].predict( non_npx[ dc_corr_feats ] ), index = npx.index, name = bmkr )
            corrected_npx[ bmkr ] = npx[ bmkr ] - npx_preds
        if save_file_name is not None:
            model_params = [ dc_corr_models[ bmkr ].intercept_ ] + list( dc_corr_models[ bmkr ].coef_ )
            add_df = pd.DataFrame( dc_corr_models[ bmkr ] ).T
            add_df.columns = [ 'const_coeff' ]+[ i + "_coeff" for i in dc_corr_feats ]
            add_df.rename( index = { 0: bmkr }, inplace = True )
            demoadj_df = demoadj_df.append( add_df )

    # End of the biomarker loop stuff...
    if save_file_name is not None:
        #Assign column-headers to dataframe and save to csv
        demoadj_df.to_csv( save_file_name ) 

    if verbose:
        print( "Done." )

    return corrected_npx, dc_corr_models

def create_model( model_type, config = None ):
    """
    Create a clean copy of a model specified by `model_type`. The hyperparameters are hard-coded in
    this function for now, but will need to be passed into this function somehow, probably in a
    dictionary that will have to be checked for each model type.
    Documentation for supported models:
     1.          Ordinary Least Sq. (OLS): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
     2.                        L1 (Lasso): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
     3.                        L2 (Ridge): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
     4.               Elastic Net (ElNet): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
     5.                             Huber: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
     6.          Stoch. Grad. Desc. (SGD): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
     7.            Random Forest (RndFor): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
     8.                  AdaBoost(AdaBst): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
     9.          Support Vec. Regr. (SVR): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    10.     Logistic Regression (LogRegr): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    11. Linear Discriminant Analyis (LDA): https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
    12.      Rand. Forest Cl. (randforcl): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    13.          Support Vector Cl. (SVC): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    14.     Decision Tree Cl. (DecTreeCl): https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    15.    Grad. Boosted Tree Cl. (gbtcl): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    16.      StackingClassifier (StackCl): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
    17.   GradientBoostingRegressor (gbr): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    18.                  PoissonRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
    Inputs:
        model_type - python string specifying the model architecture we are creating.
            config - python dictionary containing key/value pairs corresponding to the 
                     configuration of the model we're creating.
    Output:
        model object from some framework ready to be trained.
    """
    # A list of strings describing all the models this function knows about.
    all_models = [ "OLS", "Lasso", "Ridge", "ElNet", "Huber", "SGD", "RandFor", "AdaBst", "SVR", \
                   "LogRegr", "LDA", "RandForCl", "SVC", "DecTreeCl", "GBTCl", "StackCl", "GBR", \
                   "Poisson" ]
    # Find the model specified in model_type.
    if model_type.lower() == "list":
        return all_models
    # Regressors!
    elif model_type.lower() == "ols":
        if config is not None:
            return LinearRegression( **{ param: val for param, val in config.items() \
                                        if param != "model_type" } )
        else:
            return LinearRegression()
    elif model_type.lower() == "lasso":
        # If there is a configuration dictionary, use that...
        if config is not None:
            return Lasso( **{ param: val for param, val in config.items() \
                              if param != "model_type" }, random_state = my_random_seed )
        # ...otherwise, just hand back a model with some basic default parameters.
        else:
            return Lasso()
    elif model_type.lower() == "ridge":
        if config is not None:
            return Ridge( **{ param: val for param, val in config.items() \
                              if param != "model_type" }, random_state = my_random_seed )
        else:
            return Ridge()
    elif model_type.lower() == "elnet":
        if config is not None:
            return ElasticNet( **{ param: val for param, val in config.items() \
                                   if param != "model_type" } )
        else:
            return ElasticNet()
    elif model_type.lower() == "huber":
        if config is not None:
            return HuberRegressor( **{ param: val for param, val in config.items() \
                                       if param != "model_type" } )
        else:
            return HuberRegressor()
    elif model_type.lower() == "randfor":
        if config is not None:
            return RandomForestRegressor( **{ param: val for param, val in config.items() \
                                              if param != "model_type" }, \
                                          random_state = my_random_seed )
        else:
            return RandomForestRegressor()
    elif model_type.lower() == "adabst":
        if config is not None:
            return AdaBoostRegressor( **{ param: val for param, val in config.items() \
                                          if param != "model_type" } )
        else:
            return AdaBoostRegressor()
    elif model_type.lower() == "sgd":
        if config is not None:
            return SGDRegressor( **{ param: val for param, val in config.items() \
                                     if param != "model_type" } )
        else:
            return SGDRegressor()
    elif model_type.lower() == "svr":
        if config is not None:
            return SVR( **{ param: val for param, val in config.items() \
                            if param != "model_type" } )
        else:
            return SVR()
    # Classifiers!
    elif model_type.lower() == "logregr":
        if config is not None:
            return LogisticRegression( **{ param: val for param, val in config.items() \
                                           if param != "model_type" } )
        else:
            return LogisticRegression()
    elif model_type.lower() == "lda":
        if config is not None:
            return LinearDiscriminantAnalysis( **{ param: val for param, val in config.items() \
                                                   if param != "model_type" } )
        else:
            return LinearDiscriminantAnalysis()
    elif model_type.lower() == "randforcl":
        if config is not None:
            return RandomForestClassifier( **{ param: val for param, val in config.items() \
                                               if param != "model_type" }, \
                                           random_state = my_random_seed )
        else:
            return RandomForestClassifier()
    elif model_type.lower() == "svc":
        if config is not None:
            return SVC( **{ param: val for param, val in config.items() \
                            if param != "model_type" } )
        else:
            return SVC()
    elif model_type.lower() == "dectreecl":
        if config is not None:
            return DecisionTreeClassifier( **{ param: val for param, val in config.items() \
                                               if param != "model_type" } )
        else:
            return DecisionTreeClassifier()
    elif model_type.lower() == "gbtcl":
        if config is not None:
            return GradientBoostingClassifier( **{ param: val for param, val in config.items() \
                                                   if param != "model_type" } )
        else:
            return GradientBoostingClassifier()
    elif model_type.lower() == "stackcl":
        if config is not None:
            return StackingClassifier( **{ param: val for param, val in config.items() \
                                           if param != "model_type" } )
        else:
            return StackingClassifier( estimators = [ ( "lr", LogisticRegression() ), \
                                                      ( "rf", RandomForestClassifier() ) ], \
                                       final_estimator = None, cv = None, stack_method = "auto", \
                                       n_jobs = -1, passthrough = False )
    elif model_type.lower() == 'gbr':
        if config is not None:
            return GradientBoostingRegressor( **{ param: val for param, val in config.items() \
                                                  if param != "model_type" } )
        else:
            return GradientBoostingRegressor()
    elif model_type.lower() == "poisson":
        if config is not None:
            return PoissonRegressor( **{ param: val for param, val in config.items() \
                                         if param != "model_type" } )
        else:
            return PoissonRegressor()
    else:
        print( "ERROR: Model description '%s' not recognized!" % model_type )
        print( "Please choose from:" *all_models )
        return None
    
def train_model( model, features, labels, weights = None, progress = None, scaling = True ):
    """
    Train `model` using the sklearn-like `.fit()` method using `features` and `labels`.
    Inputs:
        model - untrainied model object. This is the actual model object you want to train.
        features - np.array (N x M, where N is the number of rows/measurements and M is the number 
                   of features) containing the training features.
        labels - np.array of length N containing the training labels.
        weights - np.array of length N containing the training weights (default = None).
        progress - tqdm progress bar object object (default = None).
        scaling - bool that turns on the use prescaling as implemented in 
                  `sklearn.preprocessing.scale`.
    Output:
        model : trainied model object.
    """
    # Isolate nan rows in the feature matrix, label (and the weights, if we have them) vector...
    feature_nans = np.isnan( features ).any( axis = 1 )
    label_nans = np.isnan( labels )
    if weights is not None:
        weight_nans = np.isnan( weights )
        weight_vector = weights[ ( ~feature_nans ) & ( ~label_nans ) & ( ~weight_nans ) ]
    else:
        weight_nans = np.zeros_like( labels ).astype( bool )
        weight_vector = None
    # Train the model if we are scaling and there are weights.
    if ( scaling == True ) and ( weight_vector is not None ):
        model.fit( scale( features[ ( ~feature_nans ) & ( ~label_nans ) & ( ~weight_nans ) ] ), \
                            labels[ ( ~feature_nans ) & ( ~label_nans ) & ( ~weight_nans ) ], \
                   sample_weight = weight_vector )
    # ...or if we are NOT scaling, and there are weights
    elif ( scaling == False ) and ( weight_vector is not None ):
        model.fit( features[ ( ~feature_nans ) & ( ~label_nans ) & ( ~weight_nans ) ], \
                     labels[ ( ~feature_nans ) & ( ~label_nans ) & ( ~weight_nans ) ], \
                   sample_weight = weight_vector )
    # ...or if we are scaling and there are NO weights.
    elif ( scaling == True ) and ( weight_vector is None ):
        model.fit( scale( features[ ( ~feature_nans ) & ( ~label_nans ) ] ), \
                            labels[ ( ~feature_nans ) & ( ~label_nans ) ] )
    # ...or if we are NOT scaling and there are NO weights.
    elif ( scaling == False ) and ( weight_vector is None ):
        model.fit( features[ ( ~feature_nans ) & ( ~label_nans ) ], \
                     labels[ ( ~feature_nans ) & ( ~label_nans ) ] )
    else:
        print( "Unrecognized scaling and weighting." )
        exit( -1 )
    # If we have a progress bar, update it.
    if progress is not None:
        progress.update( 1 )
    # Return the newly trained model.
    return model

def get_feature_importance_from_permutations( model, feature_data, labels, n_rounds = 10, classification = False, metric = "r2", softmax = True ):
    """
    This is intended to be a replacement in this analysis for the `extract_feature_importance_from_permutations` 
    function in `octamodel`.  The existing funciton works fine for regression problems because both `predict` 
    function spits out values that can be directly compared to the labels since they are both continuous 
    variables. For classification problems, `predict` generates binary labels to be compared with the ground 
    truth binary labels. This certainly gives you valid results, but it supresses the importance of any 
    features that can't quite flip a predicted label away from its baseline value. This would be fine if we 
    only cared about the predictions, but we regularly use the probabilities associated with those predictions 
    to generate scores. We can't just pass `predict_proba` instead because it's return value is a N x 2 matrix, 
    not a vector of length N (it returns both the positive and negative probabilities for each row in the 
    feature matrix). To deal with this, we need to write our own version of 
    `mlxtend.evaluate.feature_importance_permutation` that handles classification models in a slightly more 
    subtle way.
    
    Parameters
    ----------

    model : model object in sklearn or other package that contains the funcitons `predict` and 
            `predict_proba`
        Trained model to be evaluated.
    
    feature_data : pandas DataFrame
        DataFrame object with N rows and M columns containing all the features you are using to evaluate 
        the model. It's OK to have nans in this object (or in the labels, for that matter...) because 
        we're going to deal with nans in this funciton.

    labels : pandas Series
        Series object with N rows and the same indexing as `feature_data` that contains the lables to 
        be used as a reference point for feature importance calculations.

    n_rounds : positive integer (default: 10)
        Number of random shuffles to be done in evaluating the importance of each feature column.

    classification : boolean (default: False)
        Flag specifying whether the model being evaluated is a classifier or not (i.e. a regressor).

    metric : string (default: "r2")
        String specifying what metric we are going to examine for changes in order to extract feature 
        importance values. This clearly has to be picked from a list
    
    softmax : boolean (default: True)
        Flag specifying whether to normalize the feature importance vectors for each round so that 
        their elements sum to 1 like in a softmax function.

    Returns
    -------
    feature_importance : numpy array
        Array of n_rounds rows and M columns containing the (maybe softmaxed) feature importance values 
        calcualted by shuffling the values each column n_rounds times and checking the change in `metric`.
    
    """
    # Identify the metrics we know how to manage and check that we are doing one of them.
    allowed_metrics = { "r2": "Pearson's R^2", "auroc": "Area under the ROC curve", "rmse": "Root-mean-square error" }
    if metric not in list( allowed_metrics.keys() ):
        print( "Metric '{:}' not recognized. Please choose from:".format( metric ) )
        for metr_name, metr_desc in allowed_metrics.items():
            print( "\t{:>5}: {:}".format( metr_name, metr_desc ) )
        return None
    # Check that the metric we've been passed makes sense with the choice of classification or not. Don't throw an error, just a warning... For now.
    if classification and ( metric in [ "r2" ] ):
        print( "Metric '{:}' not usually used for classification models. You should probably choose a different one.".format( metric ) )
    if ( not classification ) and ( metric in [ "auroc" ] ):
        print( "Metric '{:}' not usually used for regression models. You should probably choose a different one.".format( metric ) )
    # Identify the rows in either the feature matrix or label vector.
    feature_nans = feature_data.isna().any( axis = 1 )
    label_nans = labels.isna()
    # Calculate the reference predictions and metric.
    if classification:
        ref_preds = pd.Series( model.predict_proba( feature_data.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ].values.astype( np.float64 ) )[ :, 1 ], \
                                                    index = feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] )
    else:
        ref_preds = pd.Series( model.predict( feature_data.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ].values.astype( np.float64 ) ), \
                                              index = feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] )
    # Scramble one feature at a time n_rounds times.
    eval_features = { shuffle_feat: [ pd.DataFrame( { feat: np.random.permutation( feature_data.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ][ feat ].values ) \
                                                      if feat == shuffle_feat else \
                                                            feature_data.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ][ feat ].values \
                                                      for feat in feature_data.columns }, \
                                                    index = feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ) for i_round in range( n_rounds ) ] \
                      for shuffle_feat in feature_data.columns }
    # Make predictions with each of the feature sets we just generated.
    eval_preds = { feat: [ pd.Series( model.predict_proba( feat_matr.values.astype( np.float64 ) )[ :, 1 ], \
                                      index = feat_matr.index )\
                           for feat_matr in eval_features[ feat ] ] \
                         if classification else \
                         [ pd.Series( model.predict( feat_matr.values.astype( np.float64 ) ), \
                                      index = feat_matr.index ) \
                           for feat_matr in eval_features[ feat ] ]
                   for feat in feature_data.columns }
    # Now, use the predictions (both reference and evaluation) to calculate the metric for each feature permutation round.
    if metric == "r2":
        scores = { "ref": linregress( labels.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                   ref_preds.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ] ).rvalue ** 2. }
        for eval_feat in feature_data.columns:
            scores[ eval_feat ] = np.array( [ linregress( labels.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                                           preds.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ] ).rvalue ** 2. \
                                              for preds in eval_preds[ eval_feat ] ] )
    elif metric == "auroc":
        scores = { "ref": roc_auc_score( labels.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                      ref_preds.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ] ) }
        for eval_feat in feature_data.columns:
            scores[ eval_feat ] = np.array( [ roc_auc_score( labels.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                                              preds.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ] ) \
                                              for preds in eval_preds[ eval_feat ] ] )
    elif metric == "rmse":
        scores = { "ref": mean_squared_error( labels.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                           ref_preds.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                              squared = False ) }
        for eval_feat in feature_data.columns:
            scores[ eval_feat ] = np.array( [ mean_squared_error( labels.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                                                   preds.loc[ feature_data.index[ ( ~feature_nans.values ) & ( ~label_nans.values ) ] ], \
                                                                  squared = False ) \
                                              for preds in eval_preds[ eval_feat ] ] )
    # Glue the scires together into an array,
    feat_imps = np.concatenate( [ np.abs( scores[ eval_feat ] - scores[ "ref" ] ).reshape( ( n_rounds, 1 ) )\
                                  for eval_feat in feature_data.columns ], axis = 1 )
    # If we are softmaxing the importances, do that.
    if softmax:
        feat_imps = feat_imps / feat_imps.sum( axis = 1 ).reshape( n_rounds, 1 )
    # All done. Return the feature importance values.
    return feat_imps

# Functions.
def get_roc( values, labels ):
    """
    Construct the Receiver Operating Characteristic (ROC) curve and calculate the area under it
    for using the elements of `values` to separate the binary states in `labels`.

    Parameters
    ----------

    values : np.array
        Model output to be tested.

    labels : np.array
        True values against which we are testing the model output.

    Returns
    -------
    dictionary
        return pacyload of a bunch of model performance metrics: AUROC, ROC parameters (false/true
        positive values and the thresholds to generate them), accuracy, precision, and recall;
    """
    # Tag the nan-valued rows in each vector (both values and lables).
    val_nans, lab_nans = np.isnan( values ), np.isnan( labels )
    # Get the ROC curve...
    fals_pos, true_pos, thresh_vals = roc_curve( labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                                                 values[ ( ~val_nans ) & ( ~lab_nans ) ] )
    predictions = np.zeros_like( values )
    predictions[ values >= 0.5 ] = 1.0
    # Calculate the AUROC.
    try:
        auroc = roc_auc_score( labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                               values[ ( ~val_nans ) & ( ~lab_nans ) ] )
        # While we're in here, let's also calculate the accuracy, precision, and recall.
        accuracy = accuracy_score( labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                              predictions[ ( ~val_nans ) & ( ~lab_nans ) ] )
        precision = precision_score( labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                                predictions[ ( ~val_nans ) & ( ~lab_nans ) ], \
                                     zero_division = 0. )
        recall = recall_score( labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                          predictions[ ( ~val_nans ) & ( ~lab_nans ) ] )
    except:
        print( "Labels:", labels[ ( ~val_nans ) & ( ~lab_nans ) ] )
        print( "Values:", values[ ( ~val_nans ) & ( ~lab_nans ) ] )
        auroc = 0.5
    # If the AUROC is below 0.5, then it's possibly still a good classifier, just with the labels
    # flipped. So in that case, let's flip the labels and re-run the AUC.
    if auroc < 0.5:
        fals_pos, true_pos, thresh_vals = roc_curve( 1.0 - labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                                                           values[ ( ~val_nans ) & ( ~lab_nans ) ] )
        auroc = roc_auc_score( 1.0 - labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                                     values[ ( ~val_nans ) & ( ~lab_nans ) ] )
        accuracy = accuracy_score( 1.0 - labels[ ( ~val_nans ) & ( ~lab_nans ) ], \
                                    predictions[ ( ~val_nans ) & ( ~lab_nans ) ] )
        precision = precision_score( 1.0 - labels[ ( ~val_nans ) & ( ~lab_nans ) ],
                                      predictions[ ( ~val_nans ) & ( ~lab_nans ) ],
                                     zero_division = 0. )
        recall = recall_score( 1.0 - labels[ ( ~val_nans ) & ( ~lab_nans ) ],
                                predictions[ ( ~val_nans ) & ( ~lab_nans ) ] )
    # Package up and return the different information we'll need to plot these ROC curves.
    return { "auroc": auroc,
          "fals_pos": fals_pos,
          "true_pos": true_pos,
       "thresh_vals": thresh_vals,
          "accuracy": accuracy,
         "precision": precision,
            "recall": recall }


def aggregate_rocs( rocs ):
    """
    Take a dictionary with keys for the true and false positives from a bunch of ROC curves whose
    values are a list of the true and false positive values respectively of these ROC curves, and
    interpolate the true posiitive values so that they are all based off of the same false
    positive values, and can therefore be plotted together with a pyplot fill_between or something
    like that.

    Parameters
    ----------

    rocs : dict
        Has three keys: "true_positives" and "false_positives" whose values are a list of true and
        false positive values from an ensemble of ROC curves, and "auroc" whose values are a list
        of the AUROCs for each of the ROCs.

    Returns
    -------
    fp_interp_vals : numpy.array
        array with sorted values from 0 to 1 in steps of 0.05.

    mean and standard deviation of tp_interp_vals evaluated at each value of fp_interp_vals

    np.array of all the ROCs in rocs.
    """
    # Parse all the true and false positive values in our ROC dictionaries.
    # Create an interpolation object for each of these TP/FP pairs.
    tp_interps = [ interpolate.interp1d( fp, tp )
                   for tp, fp in zip( rocs[ "true_positives" ], \
                                      rocs[ "false_positives" ] ) ]
    # Use each of these interpolation objects to pick interpolated true positive values for evenly
    # spaced false positive values between 0 and 1.
    fp_interp_vals = np.linspace( 0.0, 1.0, 21 )
    tp_interp_vals = np.array( [ tp_interp( fp_interp_vals ) \
                                 for tp_interp in tp_interps ] )
    # Return the evenly spaced false positive values, as well as the mean and standard deviation
    # across all the the columns of the matrix we just made. Tack the array of AUROCs on to the
    # end because that's useful too.
    return { "false_positives": fp_interp_vals,
         "true_positives_mean": tp_interp_vals.mean( axis = 0 ),
         "true_positives_stdv": tp_interp_vals.std(  axis = 0 ),
                      "aurocs": np.array( rocs[ "auroc" ] ) }