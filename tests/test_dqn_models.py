import numpy as np
import torch

from chloe.models.dqn_baseline_models import (
    BaselineCatDQNModel,
    BaselineDQNModel,
    BaselineR2D1DQNModel,
)
from chloe.models.dqn_proba_models import (
    MixedCatDQNModel,
    MixedDQNModel,
    MixedR2D1DQNModel,
)

inf_val = torch.finfo().min

class TestModels(object):
    def test_dqn_model_min_decision_turns(self):
        include_turns_in_state = True
        turn = 1 if include_turns_in_state else 0
        num_pathos = 5
        num_symptoms = 9
        num_demo = 4
        input_size = turn + num_demo + num_symptoms
        hidden_sizes = [20, 20]
        pi_hidden_sizes = [20, 20]
        lstm_size = 15
        output_size = num_symptoms + num_pathos

        max_turns = 6

        min_turns_ratio_for_decision = 3 / max_turns
        use_turn_just_for_masking = True

        input_size -= 1 if use_turn_just_for_masking else 0

        data = [0] * num_demo + [0, 1, 1, 0, -1, -1, 0, 0, 1]
        inquired_symptoms = [1, 2, 4, 5, 8]
        non_inquired_symptoms = [0, 3, 6, 7]

        # turns is equal to 0
        data1 = [0 / max_turns] + data
        # turns is equal to 3
        data2 = [3 / max_turns] + data

        model_dqn = BaselineDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            include_turns_in_state=include_turns_in_state,
            min_turns_ratio_for_decision=min_turns_ratio_for_decision,
            use_turn_just_for_masking=use_turn_just_for_masking,
        )

        model_cat_dqn = BaselineCatDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            include_turns_in_state=include_turns_in_state,
            min_turns_ratio_for_decision=min_turns_ratio_for_decision,
            use_turn_just_for_masking=use_turn_just_for_masking,
            n_atoms=51,
        )

        model_r2d1_dqn = BaselineR2D1DQNModel(
            input_size,
            hidden_sizes,
            output_size,
            lstm_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            include_turns_in_state=include_turns_in_state,
            min_turns_ratio_for_decision=min_turns_ratio_for_decision,
            use_turn_just_for_masking=use_turn_just_for_masking,
        )

        model_mixed_dqn = MixedDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            pi_hidden_sizes=pi_hidden_sizes,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            include_turns_in_state=include_turns_in_state,
            min_turns_ratio_for_decision=min_turns_ratio_for_decision,
            use_turn_just_for_masking=use_turn_just_for_masking,
        )

        model_mixed_cat_dqn = MixedCatDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            pi_hidden_sizes=pi_hidden_sizes,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            include_turns_in_state=include_turns_in_state,
            min_turns_ratio_for_decision=min_turns_ratio_for_decision,
            use_turn_just_for_masking=use_turn_just_for_masking,
            n_atoms=51,
        )

        model_mixed_r2d1_dqn = MixedR2D1DQNModel(
            input_size,
            hidden_sizes,
            output_size,
            lstm_size,
            num_symptoms,
            pi_hidden_sizes=pi_hidden_sizes,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            include_turns_in_state=include_turns_in_state,
            min_turns_ratio_for_decision=min_turns_ratio_for_decision,
            use_turn_just_for_masking=use_turn_just_for_masking,
        )
        with torch.no_grad():
            tensor1 = torch.tensor([data1])
            tensor2 = torch.tensor([data2])

            # test model_dqn for data with turns less than min_turns_ratio_for_decision
            x = model_dqn(tensor1)
            x = x.numpy()
            assert np.all(x[0, num_symptoms:] == inf_val)
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_dqn for data with turns geq than min_turns_ratio_for_decision
            x = model_dqn(tensor2)
            x = x.numpy()
            assert np.all(x[0, num_symptoms:] != -np.inf)
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_mixed_dqn: data w/ turns less than min_turns_ratio_for_decision
            x, pi = model_mixed_dqn(tensor1)
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(x[0, num_symptoms:] == inf_val)
            assert np.all(pi != -np.inf)
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_mixed_dqn: data w/ turns geq than min_turns_ratio_for_decision
            x, pi = model_mixed_dqn(tensor2)
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(x[0, num_symptoms:] != -np.inf)
            assert np.all(pi != -np.inf)
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_cat_dqn: data w/ turns less than min_turns_ratio_for_decision
            x = model_cat_dqn(tensor1)
            x = x.numpy()
            assert np.all(x[0, num_symptoms:, 0] == 1)
            assert np.all([x[0, i, 0] == 1 for i in inquired_symptoms])
            assert np.all([x[0, i, 0] != 1 for i in non_inquired_symptoms])

            # test model_cat_dqn: data w/ turns geq than min_turns_ratio_for_decision
            x = model_cat_dqn(tensor2)
            x = x.numpy()
            assert np.all(x[0, num_symptoms:, 0] != 1)
            assert np.all([x[0, i, 0] == 1 for i in inquired_symptoms])
            assert np.all([x[0, i, 0] != 1 for i in non_inquired_symptoms])

            # test model_mixed_cat_dqn: data w/ turns less min_turns_ratio_for_decision
            x, pi = model_mixed_cat_dqn(tensor1)
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(x[0, num_symptoms:, 0] == 1)
            assert np.all(pi != -np.inf)
            assert np.all([x[0, i, 0] == 1 for i in inquired_symptoms])
            assert np.all([x[0, i, 0] != 1 for i in non_inquired_symptoms])

            # test model_mixed_cat_dqn: data w/ turns geq min_turns_ratio_for_decision
            x, pi = model_mixed_cat_dqn(tensor2)
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(x[0, num_symptoms:, 0] != 1)
            assert np.all(pi != -np.inf)
            assert np.all([x[0, i, 0] == 1 for i in inquired_symptoms])
            assert np.all([x[0, i, 0] != 1 for i in non_inquired_symptoms])

            # test r2d1_dqn for data with turns less than min_turns_ratio_for_decision
            x, _ = model_r2d1_dqn(tensor1.unsqueeze(0).expand(3, -1, -1))
            x = x.numpy()
            assert np.all(x[:, 0, num_symptoms:] == inf_val)
            assert np.all([x[:, 0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[:, 0, i] != -np.inf for i in non_inquired_symptoms])

            # test r2d1_dqn for data with turns geq than min_turns_ratio_for_decision
            x, _ = model_r2d1_dqn(tensor2.unsqueeze(0).expand(3, -1, -1))
            x = x.numpy()
            assert np.all(x[:, 0, num_symptoms:] != -np.inf)
            assert np.all([x[:, 0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[:, 0, i] != -np.inf for i in non_inquired_symptoms])

            # test mixed_r2d1_dqn: data w/ turns less than min_turns_ratio_for_decision
            x, pi, _ = model_mixed_r2d1_dqn(tensor1.unsqueeze(0).expand(3, -1, -1))
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(x[:, 0, num_symptoms:] == inf_val)
            assert np.all(pi != -np.inf)
            assert np.all([x[:, 0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[:, 0, i] != -np.inf for i in non_inquired_symptoms])

            # test mixed_r2d1_dqn: data w/ turns geq than min_turns_ratio_for_decision
            x, pi, _ = model_mixed_r2d1_dqn(tensor2.unsqueeze(0).expand(3, -1, -1))
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(x[:, 0, num_symptoms:] != -np.inf)
            assert np.all(pi != -np.inf)
            assert np.all([x[:, 0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[:, 0, i] != -np.inf for i in non_inquired_symptoms])

    def test_dqn_model(self):
        turn = 0
        num_pathos = 5
        num_symptoms = 9
        num_demo = 4
        input_size = turn + num_demo + num_symptoms
        hidden_sizes = [20, 20]
        pi_hidden_sizes = [20, 20]
        lstm_size = 15
        hierarchical_map = {
            7: [3, 6],
        }
        output_size = num_symptoms + num_pathos

        data = [0] * num_demo + [0, 1, 1, 0, -1, -1, 0, 0, 1]
        inquired_symptoms = [1, 2, 4, 5, 8]
        non_inquired_symptoms = [0, 3, 6, 7]
        non_inquired_hierachical_symptoms = [0, 7]
        invalid_non_inquired_hierarchical_symptoms = [3, 6]

        model_dqn = BaselineDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
        )

        model_hierarchy_dqn = BaselineDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            hierarchical_map=hierarchical_map,
        )

        model_cat_dqn = BaselineCatDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            n_atoms=51,
        )

        model_r2d1_dqn = BaselineR2D1DQNModel(
            input_size,
            hidden_sizes,
            output_size,
            lstm_size,
            num_symptoms,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
        )

        model_mixed_dqn = MixedDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            pi_hidden_sizes=pi_hidden_sizes,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
        )

        model_mixed_cat_dqn = MixedCatDQNModel(
            input_size,
            hidden_sizes,
            output_size,
            num_symptoms,
            pi_hidden_sizes=pi_hidden_sizes,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
            n_atoms=51,
        )

        model_mixed_r2d1_dqn = MixedR2D1DQNModel(
            input_size,
            hidden_sizes,
            output_size,
            lstm_size,
            num_symptoms,
            pi_hidden_sizes=pi_hidden_sizes,
            nonlinearity=torch.nn.ReLU,
            dueling=False,
            dueling_fc_sizes=None,
            embedding_dict=None,
            freeze_one_hot_encoding=True,
            mask_inquired_symptoms=True,
            not_inquired_value=0,
            symptom_2_observation_map=None,
            patho_severity=None,
        )
        with torch.no_grad():
            tensor = torch.tensor([data])

            # test model_dqn for previously inquired vs non inquired symptoms
            x = model_dqn(tensor)
            x = x.numpy()
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_hierarchy_dqn for previously inquired vs non inquired symptoms
            x = model_hierarchy_dqn(tensor)
            x = x.numpy()
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all(
                [x[0, i] != -np.inf for i in non_inquired_hierachical_symptoms]
            )
            assert np.all(
                [x[0, i] == inf_val for i in invalid_non_inquired_hierarchical_symptoms]
            )

            # test model_mixed_dqn for previously inquired vs non inquired symptoms
            x, pi = model_mixed_dqn(tensor)
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(pi != -np.inf)
            assert np.all([x[0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_cat_dqn for previously inquired vs non inquired symptoms
            x = model_cat_dqn(tensor)
            x = x.numpy()
            assert np.all([x[0, i, 0] == 1 for i in inquired_symptoms])
            assert np.all([x[0, i, 0] != 1 for i in non_inquired_symptoms])

            # test model_mixed_cat_dqn for previously inquired vs non inquired symptoms
            x, pi = model_mixed_cat_dqn(tensor)
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(pi != -np.inf)
            assert np.all([x[0, i, 0] == 1 for i in inquired_symptoms])
            assert np.all([x[0, i, 0] != 1 for i in non_inquired_symptoms])

            # test model_r2d1_dqn for previously inquired vs non inquired symptoms
            x, _ = model_r2d1_dqn(tensor.unsqueeze(0).expand(3, -1, -1))
            x = x.numpy()
            assert np.all([x[:, 0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[:, 0, i] != -np.inf for i in non_inquired_symptoms])

            # test model_mixed_r2d1_dqn for previously inquired vs non inquired symptoms
            x, pi, _ = model_mixed_r2d1_dqn(tensor.unsqueeze(0).expand(3, -1, -1))
            x = x.numpy()
            pi = pi.numpy()
            assert np.all(pi != -np.inf)
            assert np.all([x[:, 0, i] == inf_val for i in inquired_symptoms])
            assert np.all([x[:, 0, i] != -np.inf for i in non_inquired_symptoms])
