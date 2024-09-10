config = {'datapath': 'D:/Users/david/Documents/PycharmProjects/mpr/dof/',
          'preprocess_result_path': './prep_result/',
          'fname_train'  : 'desc_trn.npy',
          'fname_test'   : 'desc_tst.npy',
          'fname_classes': 'classes.npy',
          'fname_results' : 'results.npy',
          'n_stages' : 5,
          'n_dim': 230,
          'kittle and young' : True, # feature extraction based on first order
          'save' : False,
          'n_components': 10,
          'n_selvar': 5,
          'float_limit': 1e-15,
          'n_gpu': 0,
          'worker_pool_size': 4,
          'run_type': 'build',
          'n_worker_preprocessing': None}

logreg_grid = {
                'class_weight' : ['balanced', None],
                'penalty' : ['l2', 'l1'],
                'C' : [0.001, 0.01, 0.08, 0.1, 0.15, 1.0, 10.0, 100.0],
              }
ADA_grid = {
                'n_estimators': [5, 10, 20, 50, 100],
                'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]
                 }

ETC_grid = {
            'n_estimators': [10, 50, 100, 1000],
            'max_depth': [None, 3, 5, 15]
           }

RFC_grid = {
            'n_estimators': [10, 50, 100, 1000],
            'max_depth': [None, 3, 5, 15]
           }

SVM_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'kernel': ['linear', 'poly', 'rbf'],
           }

kNN_grid = {
            'n_neighbors': [2, 3, 5, 10, 20],
            'weights': ['uniform', 'distance'],
            'leaf_size': [5, 10, 30]
          }

nBayes_grid = {
                'alpha': [0.0001, 1, 2, 10]
              }

SGD_grid = {
            'loss': ['log', 'modified_huber'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.001, 0.01],
            'l1_ratio': [0, 0.15, 0.5, 1.0],
            'learning_rate': ['optimal', 'invscaling', 'adaptive']
           }

clsfrs_params = {
            "LinearDiscriminantAnalysis1" : {},
            "LinearDiscriminantAnalysis2" : {},
            "GaussianNB" : {},
            "CalibratedClassifierCV1" : {},
            "CalibratedClassifierCV2" : {},
            # "ExtraTreesClassifier" : ETC_grid,
            "LogisticRegression": logreg_grid,
            "AdaBoostClassifier" : ADA_grid,
            "RandomForestClassifier" : RFC_grid,
            "svm.SVC": SVM_grid,
            "KNeighborsClassifier": kNN_grid,
            "nBayes_grid" : nBayes_grid,
            "SGDClassifier" : SGD_grid
         }

clsfrs = {
          "LinearDiscriminantAnalysis1" : "LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')",
          "LinearDiscriminantAnalysis2" : "LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')",
          # Create the knn model.
          # naive bayes
          "GaussianNB" : "GaussianNB()",
          "CalibratedClassifierCV1" : "CalibratedClassifierCV( LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), cv=2, method='sigmoid' )",
          "CalibratedClassifierCV2" : "CalibratedClassifierCV( LogisticRegression(), cv=2, method='sigmoid' )",
          # "ExtraTreesClassifier" : "ExtraTreesClassifier()",
          # svms
          "LogisticRegression" : "LogisticRegression( solver='liblinear', max_iter=10000 )",
          "RandomForestClassifier" : "RandomForestClassifier()",
          "svm.SVC": "svm.SVC( shrinking=True, probability=True, gamma='auto', random_state=0)",
          "KNeighborsClassifier" : "KNeighborsClassifier()",
          "nBayes_grid" : "BernoulliNB()",
          "SGDClassifier" : "SGDClassifier(eta0=1, max_iter=1000, tol=0.0001)"
        }
