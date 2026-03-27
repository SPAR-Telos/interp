mkdir data
hf download --repo-type dataset project-telos/trajectories_test_full --local-dir data/trajectories_test_full
hf download --repo-type dataset project-telos/counterfactual-grids --local-dir data/counterfactual_grids
hf download --repo-type dataset project-telos/counterfactual-trajectories --local-dir data/counterfactual_trajectories
