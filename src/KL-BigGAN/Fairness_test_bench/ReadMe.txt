Two sets of experiment in this directory
> Real Data(CelebA)
> Gen Data (Gaussian Mixture model)




Real Data
(Sort the test data into a trainable dataset, splits the data to its respective porportions)
>gen_test_data.py 

(Run FID_Fair and L2)
>sample_test.py


Gen Data*******************************************************************
(Sort the test data into a trainable dataset, splits the data to its respective porportions) + (Sample data for visualisation)
-> Also Generates the reference data required for the unbiased reference 
>gen_toy_data.py 

(Run Reference FID_fair first with command line)
>sample_test_toy_ref.py

(Run FID_fair with command line)
>sample_test_toy.py
*Can untilise Fairness_score_run.bat for batch running*
