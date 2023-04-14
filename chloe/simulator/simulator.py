import random

import gym
import numpy as np
import pyarrow as pa
from gym import spaces
from gym.utils import seeding

from chloe.utils.sim_utils import (
    convert_patho_idx_to_patho_str,
    encode_age,
    encode_ethnicity,
    encode_race,
    encode_sex,
    load_and_check_data,
    load_record_batch,
    retrieve_file_from_shared_memory,
)

# none value - when an inquiry for a symptom is not yet done
NONE_VAL = 0
# presence value - indicating a symptom is present
PRES_VAL = 1
# absence value - indicating a symptom is absent
ABS_VAL = -1


class RewardConfig(object):
    """Class for holding config options for reward function.
    """

    def __init__(
        self,
        reward_on_repeated_action=0,
        reward_on_missing_diagnosis=-1,
        reward_on_correct_diagnosis=1.0,
        reward_on_intermediary_turns=0,
        reward_on_relevant_symptom_inquiry=0,
        reward_on_irrelevant_symptom_inquiry=0,
        reward_on_missing_atcd_recall=0,
        reward_on_missing_symptoms_recall=0,
        reward_on_reaching_max_turns=0,
        rescale_correct_diagnosis_reward_on_differential=True,
    ):
        """Init method of the reward configuration.

        Parameters
        ----------
        reward_on_repeated_action: float
            Reward obtained if the agent repeat the same action.
            Default: 0
        reward_on_missing_diagnosis: float
            Reward obtained when the agent infers a pathology which
            is not correct. Default: -1
        reward_on_correct_diagnosis: float
            Reward obtained when the agent infers a pathology which
            is correct. Default: 1
        reward_on_intermediary_turns: float or `callable`
            Reward obtained after each additional turn. only the temporal
            aspect is taken into account here. This could be float or a
            function of the form `func(t)`, with `t` being the current
            turn, returning a float. Default: 0
        reward_on_relevant_symptom_inquiry: float
            Reward obtained when the agent inquires a symptom that is relevant
            for the current disease being simulated. Default: 0
        reward_on_irrelevant_symptom_inquiry: float
            Reward obtained when the agent inquires a symptom that is not
            relevant for the current disease being simulated. Default: 0
        reward_on_missing_atcd_recall: float
            Reward to be given per antecedent missing from the
            agent-patient interaction. This reward signal encourages the
            agent to ask more antecedents related relevant questions. Default: 0
        reward_on_missing_symptoms_recall: float
            Reward to be given per symptom missing from the
            agent-patient interaction. This reward signal encourages the
            agent to ask more relevant symptoms related questions. Default: 0
        reward_on_reaching_max_turns: float
            Reward obtained when the agent reach the max number of turns. Default: 0

        """
        self.reward_on_repeated_action = reward_on_repeated_action
        self.reward_on_missing_diagnosis = reward_on_missing_diagnosis
        self.reward_on_correct_diagnosis = reward_on_correct_diagnosis
        self.reward_on_intermediary_turns = reward_on_intermediary_turns
        self.reward_on_relevant_symptom_inquiry = reward_on_relevant_symptom_inquiry
        self.reward_on_irrelevant_symptom_inquiry = reward_on_irrelevant_symptom_inquiry
        self.reward_on_missing_symptoms_recall = reward_on_missing_symptoms_recall
        self.reward_on_missing_atcd_recall = reward_on_missing_atcd_recall
        self.reward_on_reaching_max_turns = reward_on_reaching_max_turns
        self.rescale_correct_diagnosis_reward_on_differential = (
            rescale_correct_diagnosis_reward_on_differential
        )

    @staticmethod
    def htc_config_sparse_neurips18():
        return RewardConfig(
            reward_on_repeated_action=0.0,
            reward_on_missing_diagnosis=-1.0,
            reward_on_correct_diagnosis=1.0,
            reward_on_intermediary_turns=0,
            reward_on_relevant_symptom_inquiry=0,
            reward_on_irrelevant_symptom_inquiry=0,
        )

    @staticmethod
    def htc_config_sparse_aaai18():
        return RewardConfig(
            reward_on_repeated_action=-1.0,
            reward_on_missing_diagnosis=-0.0,
            reward_on_correct_diagnosis=1.0,
            reward_on_intermediary_turns=0,
            reward_on_relevant_symptom_inquiry=0,
            reward_on_irrelevant_symptom_inquiry=0,
        )

    @staticmethod
    def baidu_config_acl18():
        return RewardConfig(
            reward_on_repeated_action=0.0,
            reward_on_missing_diagnosis=-22.0,
            reward_on_correct_diagnosis=44.0,
            reward_on_intermediary_turns=-1,
            reward_on_relevant_symptom_inquiry=0,
            reward_on_irrelevant_symptom_inquiry=0,
        )


class PatientInteractionSimulator(gym.Env):
    """Class for simulating patients interaction for an RL project.

    The simulator is based on a csv file exported from a Synthea simulation.

    """

    metadata = {"render.modes": ["step", "all"]}

    def __init__(
        self,
        patient_filepath="chloe/data/symcat/simulated_patients/symptoms.csv",
        symptom_filepath="chloe/data/symcat/parsed_jsons/symptoms.json",
        condition_filepath="chloe/data/symcat/parsed_jsons/conditions.json",
        max_turns=0,
        stop_if_repeated_question=False,
        action_type=0,
        include_turns_in_state=False,
        include_race_in_state=True,
        include_ethnicity_in_state=True,
        is_reward_relevancy_patient_specific=False,
        use_differential_diagnosis=True,
        reward_config=None,
        shared_data_socket=None,
        use_initial_symptom_flag=False,
        travel_evidence="trav1",
        travel_negative_response="N",
        default_location="AmerN"
    ):
        """Init method of the simulator.

        Parameters
        ----------
        patient_filepath: str
            path to the csv file containing generated patients
            from Synthea simulator. If the `shared_data_socket` is
            specified (not None) , then it corresponds to the prefix
            to use to retrieve the data in the shared memory associated
            with that socked.

        symptom_filepath:  str
            path to a json file containing the symptom data.
            the minimum structure of the data should be:

            .. code-block:: text

                {
                    key_symptom1: {
                        'name': symptom-name1,
                        ...
                    },
                    key_symptom2: {
                        'name': symptom-name2,
                        ...
                    },
                    ...
                }

        condition_filepath:  str
            path to a json file containing the condition data.
            the minimum structure of the data should be:

            .. code-block:: text

                {
                    key_condtion1: {
                        'condition_name': condition-name1,
                        'symptoms':{
                            key_symptom1:{
                                ...
                            },
                            key_symptom8:{
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    key_condtion2: {
                        'condition_name': condition-name2,
                        'symptoms':{
                            key_symptom4:{
                                ...
                            },
                            key_symptom2:{
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    ...
                }

        max_turns: int
            the maximum number of turns allowed. If less or equals than zero,
            then an interaction session may last indefinitely. Default: 0

        stop_if_repeated_question: bool
            indicate if the interaction session should terminate if an inquiry
            regarding a symptom is made twice. Default: False

        action_type: int
            indicate the type of action to be used:
                - 0: (single) int action
                - otherwise: triplet [type_action, symptom, disease]:
                    - type_action: 0 (symptom inquiry) or 1 (diagnosis)
                    - symptom: inquired symptom
                    - disease: diagnosed disease
            Default: 0

        include_turns_in_state: bool
            indicate if the turn number should be included
            in the state. Default: False

        include_race_in_state: bool
            indicate if the race attribute should be included
            in the state. Default: True

        include_ethnicity_in_state: bool
            indicate if the ethnicity attribute should be included
            in the state. Default: True

        is_reward_relevancy_patient_specific: bool
            indicate if the relevancy component of the reward is specific
            to the patient or not. If specific to the patient, the agent is
            rewarded if it inquires about symptoms relevant for the patient
            currently simulated (we are limited to symptoms that the patient
            currently has). Otherwise, the agent is rewarded if it inquires
            about symptoms relevant for the pathology the current patient is
            suffering from (we consider all the symptoms of the pathology even
            if the patient does not have them all.). Default: False

        use_differential_diagnosis: bool
            flag indicating whether or not the simulator should make use of
            differential diagnosis present in the dataset. Default: True

        reward_config: RewardConfig
                the configuration for the reward

        shared_data_socket: str, None
                the socket to be used for retrieveing data in a shared memory.

        use_initial_symptom_flag: boolean
                A flag indicating whether to use the initial symptoms from the
                dataset instead of randomly sample it. Default: False.

        travel_evidence: str
                the evidence code corresponding to traveling activity.
                Default: trav1.

        travel_negative_response: str
                the evidence value code corresponding to a negative response to the
                traveling activity. Default: N.

        default_location: str
                the evidence value code corresponding to the default patient location.
                Default: AmerN.


        """
        super(PatientInteractionSimulator, self).__init__()

        self.filepath = patient_filepath
        self.use_initial_symptom_flag = use_initial_symptom_flag
        self.max_turns = max_turns
        self.action_type = action_type
        self.include_turns_in_state = include_turns_in_state
        self.include_race_in_state = include_race_in_state
        self.include_ethnicity_in_state = include_ethnicity_in_state
        self.is_reward_relevancy_patient_specific = is_reward_relevancy_patient_specific
        self.use_differential_diagnosis = use_differential_diagnosis
        self.shared_data_socket = shared_data_socket
        self.travel_evidence = travel_evidence
        self.travel_negative_response = travel_negative_response
        self.default_location = default_location
        if not self.max_turns and self.include_turns_in_state:
            raise ValueError(
                "'max_turns' could not be None/0 if 'include_turns_in_state' is True."
            )
        self.reward_config = reward_config
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        self.stop_if_repeated_question = stop_if_repeated_question

        # load patient data
        if self.shared_data_socket is None:
            [
                self.rb,
                self.unique_symptoms,
                self.unique_pathos,
                patho_symptoms,
                self.unique_races,
                self.unique_ethnics,
                self.unique_genders,
                self.symptoms_with_multiple_answers,
                self.max_differential_len,
                self.unique_differential_pathos,
                self.unique_init_symptoms,
            ] = load_record_batch(self.filepath)
        else:
            [
                self.rb,
                self.unique_symptoms,
                self.unique_pathos,
                patho_symptoms,
                self.unique_races,
                self.unique_ethnics,
                self.unique_genders,
                self.symptoms_with_multiple_answers,
                self.max_differential_len,
                self.unique_differential_pathos,
                self.unique_init_symptoms,
            ] = retrieve_file_from_shared_memory(self.shared_data_socket, self.filepath)

        if self.rb.num_rows == 0:
            raise ValueError("The provided file contains no valid patients.")

        # convert the patho_symptoms into string format
        patho_symptoms = {
            self.unique_pathos[a]: set(
                [self.unique_symptoms[b] for b in patho_symptoms[a]]
            )
            for a in patho_symptoms.keys()
        }

        # load and check symptoms and pathology
        self._load_and_check_symptoms_with_pathos(
            symptom_filepath,
            self.unique_symptoms,
            self.symptoms_with_multiple_answers,
            condition_filepath,
            self.unique_pathos,
            patho_symptoms,
        )

        # define evidence landscaping / evi2evi and evi2patho association matrices
        self._define_evidence_landscaping()

        # number of demographics features (age, sex, race, ethnic)
        # the first 4 entries corespond respectively to:
        #    Age[0-7], Sex[0-1], Race[0-4], Ethnicity[0-1]
        self.num_demo_features = 2
        low_demo_values = [0] * self.num_demo_features
        high_demo_values = [7, 1]
        if self.include_race_in_state:
            self.num_demo_features += 1
            low_demo_values += [0]
            high_demo_values += [4]
        if self.include_ethnicity_in_state:
            self.num_demo_features += 1
            low_demo_values += [0]
            high_demo_values += [1]
        if self.include_turns_in_state:
            low_demo_values = [0] + low_demo_values
            high_demo_values = [1] + high_demo_values
            self.num_demo_features += 1

        # define the action and observation spaces
        self.num_symptoms = len(self.symptom_index_2_key)
        self.num_pathos = len(self.pathology_index_2_key)
        if not (self.num_symptoms > 0 and self.num_pathos > 0):
            raise ValueError(
                "Either the number of symptoms or the number of pathologies is null."
            )
        self._define_action_and_observation_spaces(
            self.num_symptoms,
            self.num_pathos,
            self.num_demo_features,
            low_demo_values,
            high_demo_values,
            np.float32,
        )
        # define default symptoms values in frame
        self._define_evidence_default_value_in_frame()

        # turns in the interaction process (dialog)
        self.turns = 0
        # the index of the currenly simulated patient
        self.current_index = 0
        # the information collected so far in an interaction session
        self.frame = None
        # is the information being collected has been updated?
        self.is_symptom_updated = False
        # is the information being collected is absent for the patient at hand?
        self.is_evidence_absent = True
        # is the information being collected is relevant for the pathology?
        self.is_symptom_relevant = False
        # is the information being collected is an antecedent?
        self.is_symptom_antecedent = False
        # order of the question asked
        self.ordered_actions = []
        # order of reward provided
        self.ordered_rewards = []
        # implicit set symptoms
        self.implicit_symptoms = set()
        # implicit set symptoms that were inquired
        self.explicit_inquired_implicit_symptoms = set()

        # default patient demographics info
        self.age = None
        self.race = None
        self.sex = None
        self.ethnic = None

        # pathology to infer
        self.target_pathology = None
        self.target_pathology_index = None
        self.target_pathology_severity = None

        # inferred pathology
        self.inferred_pathology = None

        # involve the seed
        self.seed()

    def _define_action_and_observation_spaces(
        self,
        num_symptoms,
        num_pathos,
        num_demo_features,
        low_demo_values,
        high_demo_values,
        obs_dtype,
    ):
        """ Utility function for defining the enviroment action and observation spaces.

        It define the action and observation spaces for this Gym environment.

        Parameters
        ----------
        num_symptoms: int
            number of possible symptoms.
        num_pathos: int
            number of possible pathos.
        num_demo_features: int
            number of features corresponditing to demographic data.
        low_demo_values: list
            low values for demographic features.
        high_demo_values: list
            high values for demographic features.
        obs_dtype: dtype
            dtype of the observation data.

        Returns
        -------
        None
        """

        if self._has_int_action():
            num_actions = num_symptoms + num_pathos
            self.num_actions = [num_actions]
            self.action_space = spaces.Discrete(self.num_actions[0])
        else:
            self.num_actions = [2, num_symptoms, num_pathos]
            self.action_space = spaces.MultiDiscrete(self.num_actions)

        msg = "the length of low/high_demo_values must match num_demo_features."
        assert len(low_demo_values) == len(high_demo_values) == num_demo_features, msg

        # low and high values of each entry of the observation/state space
        low_val = list(low_demo_values)
        high_val = list(high_demo_values)

        # low and high values of each entry of the observation/state space
        # dedicated to symptoms
        symp_low_val = []
        symp_high_val = []

        # mapping the symptom index to the [start, end] indices in the obs data
        symptom_to_obs_mapping = {}

        # mapping the symptom index to possible values
        symptom_possible_val_mapping = {}

        # mapping the symptom index to the symptom types
        symptom_data_types = {}

        # mapping the symptom index to the symptom default value
        symptom_default_value_mapping = {}

        # integer based categorical value
        categorical_integer_symptoms = set()

        for idx in range(len(self.symptom_index_2_key)):
            key = self.symptom_index_2_key[idx]
            data_type = self.symptom_data[key].get("type-donnes", "B")
            possible_values = self.symptom_data[key].get("possible-values", [])
            default_value = self.symptom_data[key].get("default_value", None)
            start_obs_idx = len(symp_low_val) + num_demo_features
            symptom_data_types[idx] = data_type
            num_elts = len(possible_values)
            if num_elts > 0:
                assert default_value in possible_values

            if data_type == "B":
                # binary symptom
                symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
            elif data_type == "C":
                # categorical symptom
                assert num_elts > 0
                if isinstance(possible_values[0], str):
                    for k in range(num_elts):
                        symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                        symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                    symptom_to_obs_mapping[idx] = [
                        start_obs_idx,
                        start_obs_idx + num_elts,
                    ]
                else:
                    # integer value
                    categorical_integer_symptoms.add(idx)
                    symp_low_val.append(min(NONE_VAL, PRES_VAL))
                    symp_high_val.append(max(NONE_VAL, PRES_VAL))
                    symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
                symptom_possible_val_mapping[idx] = {
                    a: i for i, a in enumerate(possible_values)
                }
                symptom_default_value_mapping[idx] = default_value
            elif data_type == "M":
                # multi-choice symptom
                assert num_elts > 0
                for k in range(num_elts):
                    symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                    symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + num_elts]
                symptom_possible_val_mapping[idx] = {
                    a: i for i, a in enumerate(possible_values)
                }
                symptom_default_value_mapping[idx] = default_value
            else:
                raise ValueError(
                    f"Symptom key: {key} - Unknown data type: {data_type}."
                )

        low_val.extend(symp_low_val)
        high_val.extend(symp_high_val)
        self.obs_dtype = obs_dtype
        self.num_features = len(low_val)
        self.symptom_to_obs_mapping = symptom_to_obs_mapping
        self.symptom_possible_val_mapping = symptom_possible_val_mapping
        self.symptom_data_types = symptom_data_types
        self.symptom_default_value_mapping = symptom_default_value_mapping
        self.categorical_integer_symptoms = categorical_integer_symptoms
        self.observation_space = spaces.Box(
            low=np.array(low_val), high=np.array(high_val), dtype=self.obs_dtype
        )

    def _define_evidence_default_value_in_frame(self):
        """Get the position and values for symptom default values in frame observation.

        Parameters
        ----------

        Returns
        -------
        None

        """
        result = []
        for idx in range(len(self.symptom_index_2_key)):
            key = self.symptom_index_2_key[idx]
            data_type = self.symptom_data[key].get("type-donnes", "B")
            if data_type == "B":
                fi = self._from_symptom_index_to_frame_index(idx, None)
                result.append((fi, ABS_VAL))
            else:
                default_value = self.symptom_default_value_mapping[idx]
                fi = self._from_symptom_index_to_frame_index(idx, default_value)
                if idx in self.categorical_integer_symptoms:
                    val_index = self.symptom_possible_val_mapping[idx][default_value]
                    # scale value
                    num = len(self.symptom_possible_val_mapping[idx])
                    scaled = NONE_VAL + ((PRES_VAL - NONE_VAL) * (val_index + 1) / num)
                    result.append((fi, scaled))
                else:
                    result.append((fi, PRES_VAL))
        self.symptom_defaul_in_obs = result

    def get_symptom_to_observation_mapping(self):
        """Utility to get the index range in the state space associated to each symptom.

        Utility function to get the mapping from all symptom indices to index ranges
        associated to those symptoms in the observation space.

        Parameters
        ----------

        Returns
        -------
        result: dict
            the mapping from symptom index to index range associated to
            that symptom in the observation space.

        """
        return self.symptom_to_obs_mapping

    def get_pathology_severity(self):
        """Utility to get the severity associated to each pathology.

        Parameters
        ----------

        Returns
        -------
        result: list
            the severity associated to each pathology. The pathologies are
            identified by their index in the list.

        """
        return self.pathology_severity_data

    def get_evidence_2_pathology_association(self):
        """Utility to get the evidence to pathology association.

        Parameters
        ----------

        Returns
        -------
        result: np.array
            the evidence-pathology association.

        """
        return self.evidence_2_patho_association

    def get_evidence_2_evidence_association(self):
        """Utility to get the evidence to evidence association.

        Parameters
        ----------

        Returns
        -------
        result: np.array
            the evidence-evidence association.

        """
        return self.evidence_2_evidence_association

    def get_evidence_default_value_in_obs(self):
        """Utility to get the evidence default value in observation frame.

        Parameters
        ----------

        Returns
        -------
        result: list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing.

        """
        return self.symptom_defaul_in_obs

    def _has_int_action(self):
        """Utility to check if the actions are of type int or tuple of int.

        Parameters
        ----------

        Returns
        -------
        result: bool
            True if actions are of type int.

        """
        return self.action_type == 0

    def _has_multi_choice_symptoms(self, symptom_data):
        """Utility to check if the simulator support multi-choice symptoms.

        It checks if the the provided symptom data contains
        symptoms associated with multi-choice answers.
        E.g.: `douleurxx_endroits` which coresspond to the body parts
        where you have pain.

        Parameters
        ----------
        symptom_data: dict
            the json dict describing the symptom data provided to the
            simulator.

        Returns
        -------
        result: bool
            True if the symptom data contains multi-choice symptoms.

        """
        for k in symptom_data.keys():
            if symptom_data[k].get("type-donnes", "B") == "M":
                return True
        return False

    def get_hierarchical_symptom_dependency(self):
        """Get the groups of symptoms that depend on some master symptoms.

        Returns
        -------
        result: dict
            dictionnary of the groups of dependent symptom indices. The key of
            this dictionary is the index of the master symptom.
        """
        return self.all_linked_symptom_indices

    def _get_linked_symptoms(self, symptom_data):
        """Get the groups of symptoms that are linked together.

        Symptoms are linked together if they share the same code_question.

        Parameters
        ----------
        symptom_data: dict
            dictionary representing the loaded symptom JSON file.

        Returns
        -------
        result: dict
            dictionnary of the groups of linked symptoms. The key of
            this dictionary is the base symptom.

        """
        result = {}
        for k in symptom_data.keys():
            code_question = symptom_data[k].get("code_question", k)
            if code_question not in result:
                result[code_question] = []
            result[code_question].append(k)

        # Eliminate entries with just one element in the list as
        # those are independent questions.
        # Also, retrieve the base question (it is the one equals to code_question)
        # and eliminate it from the related list (it already serves as key)
        all_keys = list(result.keys())
        for k in all_keys:
            if len(result[k]) == 1:
                result.pop(k)
            else:
                assert k in result[k]
                result[k].remove(k)

        # return the computed map
        return result

    def _load_and_check_symptoms_with_pathos(
        self,
        symptom_filepath,
        unique_symptoms,
        symptoms_with_multiple_answers,
        condition_filepath,
        unique_pathos,
        patho_symptoms,
    ):
        """Check symptom/condition JSON file validity against the provided patient file.

        It loads the symptom/pathology data and check
        if they are compliant with the ones defined
        in the `unique_symptoms`/`unique_pathos` provided
        in the patient file.

        Parameters
        ----------
        symptom_filepath: str
            path to a json file containing the symptom data.
        unique_symptoms: list
            a list of unique symptoms within the provided patient data.
        symptoms_with_multiple_answers: list
            a list of unique symptoms with multiple answer within the provided
            patient data.
        condition_filepath: str
            path to a json file containing the pathology data.
        unique_pathos: list
            a list of unique pathologies within the provided patient data.
        patho_symptoms: dict
            a mapping from a pathology to a set of symptoms describing that
            pathology as derived from the patient data.

        Returns
        -------
        None

        """

        # load symptoms
        symptom_infos = load_and_check_data(
            symptom_filepath, unique_symptoms, key_name="name"
        )
        self.symptom_index_2_key = symptom_infos[0]
        self.symptom_name_2_index = symptom_infos[1]
        self.symptom_data = symptom_infos[2]
        self.multi_choice_flag = self._has_multi_choice_symptoms(self.symptom_data)
        self.all_symptom_names = [
            self.symptom_data[k]["name"] for k in self.symptom_data.keys()
        ]
        self.all_linked_symptoms = self._get_linked_symptoms(self.symptom_data)
        # transform 'self.all_linked_symptoms' from str into symptom indices
        self.all_linked_symptom_indices = {
            self.symptom_name_2_index[self.symptom_data[base_symp_key]["name"]]: [
                self.symptom_name_2_index[self.symptom_data[linked_symp_key]["name"]]
                for linked_symp_key in self.all_linked_symptoms[base_symp_key]
            ]
            for base_symp_key in self.all_linked_symptoms.keys()
        }
        # reverse the linked symptoms map: from linked_symptom => base_symptom
        self.all_linked_reverse_symptom_indices = {
            linked_symp_idx: base_symp_idx
            for base_symp_idx in self.all_linked_symptom_indices
            for linked_symp_idx in self.all_linked_symptom_indices[base_symp_idx]
        }
        for a in symptoms_with_multiple_answers:
            idx = self.symptom_name_2_index[a]
            key = self.symptom_index_2_key[idx]
            data_type = self.symptom_data[key].get("type-donnes", "B")
            if data_type != "M":
                raise ValueError(
                    f"Unconsistency with Symptom {a}: Occured multiple times while"
                    f" not a multiple choice symptom. {symptoms_with_multiple_answers}"
                )

        # load pathologies
        pathology_infos = load_and_check_data(
            condition_filepath, unique_pathos, key_name="condition_name"
        )
        self.pathology_index_2_key = pathology_infos[0]
        self.pathology_name_2_index = pathology_infos[1]
        self.pathology_data = pathology_infos[2]
        self.pathology_defined_symptoms = {}
        # get all pathology severity - # default severity to 0
        self.pathology_severity_data = [
            self.pathology_data[self.pathology_index_2_key[idx]].get("urgence", 0)
            for idx in range(len(self.pathology_index_2_key))
        ]

        # check if the provided df respect the symptom/patho relationships
        for key in self.pathology_data.keys():
            defined_symptom_keys = list(self.pathology_data[key]["symptoms"].keys())
            if "antecedents" in self.pathology_data[key]:
                defined_symptom_keys += list(
                    self.pathology_data[key]["antecedents"].keys()
                )
            defined_symptoms = []
            for k in defined_symptom_keys:
                symp_name = self.symptom_data[k]["name"]
                symp_type = self.symptom_data[k].get("type-donnes", "B")
                # binary symptoms
                if symp_type == "B":
                    defined_symptoms.append(symp_name)
                else:
                    # categorical or multi-choice
                    possible_values = self.symptom_data[k].get("possible-values", [])
                    for v in possible_values:
                        val_name = symp_name + "_@_" + str(v)
                        defined_symptoms.append(val_name)
            defined_symptoms = set(defined_symptoms)
            self.pathology_defined_symptoms[key] = defined_symptoms
            patho = self.pathology_data[key]["condition_name"]

            if patho in patho_symptoms:
                data_symptoms = patho_symptoms[patho]
                diff = data_symptoms - defined_symptoms
                if len(diff) > 0:
                    raise ValueError(
                        f"Unconsistency with patho {patho}: Unauthorized symptoms {diff}"
                    )

    def _define_evidence_landscaping(self):
        """Get associations between evidences and either evidences and pathologies.

        Parameters
        ----------

        Returns
        -------
        None

        """
        self.evidence_2_evidence_association = np.zeros(
            (len(self.symptom_index_2_key), len(self.symptom_index_2_key)), dtype=bool
        )
        self.evidence_2_patho_association = np.zeros(
            (len(self.symptom_index_2_key), len(self.pathology_index_2_key)), dtype=bool
        )
        for d, key in enumerate(self.pathology_index_2_key):
            defined_symptom_keys = list(self.pathology_data[key]["symptoms"].keys())
            if "antecedents" in self.pathology_data[key]:
                defined_symptom_keys += list(
                    self.pathology_data[key]["antecedents"].keys()
                )
            evi_idx = [
                self.symptom_name_2_index[self.symptom_data[evi]["name"]]
                for evi in defined_symptom_keys
            ]
            self.evidence_2_patho_association[evi_idx, d] = True
            for i in evi_idx:
                self.evidence_2_evidence_association[evi_idx, i] = True

    def seed(self, seed=None):
        """Seed for the simulator process.

        Parameters
        ----------
        seed: int, None
            Value of the seed.

        Returns
        -------
        result: list
            the list of seeds used in this env's random
            number generators.

        """
        self.np_random, seed = seeding.np_random(seed)
        self.np_random = random.Random(seed)
        return [seed]

    def get_symptom_and_value(self, symptom_name):
        """Utility function to get the symptom and the associated value from csv data.

        Given a symptom, find its root and assocaited value
        for example, `douleurxx_carac_@_penible` will return
        `douleurxx_carac` and `penible`.
        Similarly, `fievre` will return `fievre` and  None (which
        is the default value for boolean symptom).

        Parameters
        ----------
        symptom_name: str
            the symptom (from csv) for which we want to retrieve the info data.

        Returns
        -------
        symptom: str
            the symptom name as defined in the config file.
        val: object
            the value associated to the symptom.

        """
        idx = symptom_name.find("_@_")
        if idx == -1:
            # boolean symptom
            return symptom_name, None
        else:
            elem_base = symptom_name[:idx]
            elem_val = symptom_name[idx + 3 :]
            base_idx = self.symptom_name_2_index.get(elem_base, -1)

            assert base_idx != -1, (
                f"The symptom {elem_base} is not defined "
                f"while receiving {symptom_name}!"
            )

            assert self.symptom_possible_val_mapping.get(base_idx)
            if not (elem_val in self.symptom_possible_val_mapping[base_idx]):
                # convert to the right type
                elem_val = int(elem_val)
                assert elem_val in self.symptom_possible_val_mapping[base_idx]

            return elem_base, elem_val

    def _set_demo_features(self, frame, age, sex, race, ethnic):
        """Set the demographic features in the provided observation frame.

        Parameters
        ----------
        frame: np.array
            the observation frame to be updated.
        age: int
            the age of the patient.
        sex: str
            the sex of the patient.
        race: str
            the race of the patient.
        ethnic: str
            the ethnic of the patient.

        Returns
        -------
        result: np.array
            the updated observation frame.

        """
        tmp_init_idx = 1 if self.include_turns_in_state else 0

        # set the age
        frame[tmp_init_idx + 0] = encode_age(age)
        # set the sex
        frame[tmp_init_idx + 1] = encode_sex(sex)
        # set the race
        if self.include_race_in_state:
            frame[tmp_init_idx + 2] = encode_race(race)
        # set the ethnicity
        if self.include_ethnicity_in_state:
            diff = 0 if self.include_race_in_state else 1
            frame[tmp_init_idx + 3 - diff] = encode_ethnicity(ethnic)
        # return the updated frame
        return frame

    def _compute_differential_probs(self, differential):
        """Compute the differential probability from the diffential scores.

        Parameters
        ----------
        differential: dict
            Map of the pathology id in the differential to its sommeOR and score
            as returned by DXA.

        Returns
        -------
        indices: np.ndarray
            the array correspondind to the pathology indices involved in the
            differential (-1 is used for padding).
        probability: np.ndarray
            the array representing the computed probabilities associated to the
            pathologies represented by `indices`.

        """
        if differential is None or not self.use_differential_diagnosis:
            return None, None
        else:
            assert len(differential) <= self.max_differential_len
            indices = np.ones(self.max_differential_len, dtype=int) * -1
            probability = np.zeros(self.max_differential_len, dtype=np.float32)
            sumProba = 0
            for i, k in enumerate(differential.keys()):
                indices[i] = k
                sommeOR, _ = differential[k]
                proba = sommeOR / (1.0 + sommeOR)
                sumProba += proba
                probability[i] = proba
            if sumProba != 0:
                probability = probability / sumProba
            else:
                probability[0 : len(differential)] = 1.0 / len(differential)

        # sort in desceding order according to proba
        s_ind = np.argsort(probability, axis=-1)
        probability = probability[s_ind[::-1]]
        indices = indices[s_ind[::-1]]

        return indices, probability

    def _read_rb_data(self, rb_data):
        """Read the info data from pyarrow rb_data record.

        Parameters
        ----------
        rb_data: dict
            pyarraow record data.

        Returns
        -------
        age: int
            patient age.
        race: str
            patient race.
        sex: int
            patient sex.
        ethnic: str
            patient ethnic.
        pathology: str
            patient pathology.
        pathology_index: int
            index of the patient pathology.
        pathology_severity: int
            severity of the patient pathology.
        symptoms: list
            list of the symptoms experienced by the patient.
        differential_indices: np.array
            the array correspondind to the pathology indices involved in the
            differential (-1 is used for padding).
        differential_probas: np.array
            the array representing the computed probabilities associated to the
            pathologies represented by `indices`.
        initial_symptom: str
            patient initial symptom.

        """
        age = rb_data["AGE_BEGIN"][0]
        race = self.unique_races[rb_data["RACE"][0]]
        sex = self.unique_genders[rb_data["GENDER"][0]]
        ethnic = self.unique_ethnics[rb_data["ETHNICITY"][0]]
        pathology = self.unique_pathos[rb_data["PATHOLOGY"][0]]
        pathology_index = self.pathology_name_2_index[pathology]
        pathology_severity = self.pathology_severity_data[pathology_index]
        symptoms = [self.unique_symptoms[a] for a in rb_data["SYMPTOMS"][0]]
        initial_symptom = (
            None
            if len(self.unique_init_symptoms) == 0
            else self.unique_init_symptoms[rb_data["INITIAL_SYMPTOM"][0]]
        )

        # get the diffential if they are provided
        # we manually filtering out data that are not part of the simulated pathologies
        differential_data = (
            None
            if (self.max_differential_len == -1 or not self.use_differential_diagnosis)
            else {
                self.pathology_name_2_index[
                    self.unique_differential_pathos[int(diff_data[0])]
                ]: [
                    diff_data[1],
                    diff_data[2],
                ]
                for diff_data in rb_data["DIFFERNTIAL_DIAGNOSIS"][0]
                if self.unique_differential_pathos[int(diff_data[0])]
                in self.pathology_name_2_index
            }
        )
        out_diff = self._compute_differential_probs(differential_data)
        differential_indices, differential_probas = out_diff

        result = [
            age,
            race,
            sex,
            ethnic,
            pathology,
            pathology_index,
            pathology_severity,
            symptoms,
            differential_indices,
            differential_probas,
            initial_symptom,
        ]
        return result

    def _generate_mask_data(self, patho_index, target_symptoms, degree=1):
        """Generates a mask to obtain a corrupted version of the provided patient data.

        Parameters
        ----------
        patho_index: int
            the index of the considered pathology.
        target_symptoms: list
            list of symptoms that are associated with the simulated pathology.
        degree: float
            degree under which the samples from uniform distribution will be powered to
            in order to get the probability of the bernouilli distribution, i.e,
            p = (u~U(0,1)^degree). Default: 1

        Returns
        -------
        result: np.array
            the generated mask.

        """
        patho_key = self.pathology_index_2_key[patho_index]

        all_symp_keys = list(self.pathology_data[patho_key]["symptoms"].keys())
        all_binary_symp_keys = [
            a
            for a in self.pathology_data[patho_key]["symptoms"]
            if self.symptom_data_types[
                self.symptom_name_2_index[self.symptom_data[a]["name"]]
            ]
            == "B"
        ]
        all_atcd_keys = list(
            self.pathology_data[patho_key].get("antecedents", {}).keys()
        )
        all_keys = all_symp_keys + all_atcd_keys
        remainder_keys = list(set(self.symptom_data.keys()) - set(all_keys))

        first = random.sample(list(set(target_symptoms) & set(all_binary_symp_keys)), 1)
        u = random.random()
        p = u ** degree
        mask = np.random.binomial(1, p, len(all_keys))

        selected_keys = [
            all_keys[i]
            for i in range(len(all_keys))
            if mask[i] == 1 or all_keys[i] == first
        ]

        if self.max_turns:
            remain_elts = max(0, self.max_turns - len(selected_keys))
            tmp_num = min(remain_elts, len(remainder_keys))
            u = random.random()
            p = u ** (degree + 1)
            mask = np.random.binomial(1, p, max(1, tmp_num))
            num_selected = mask.sum()
            tmp_select = (
                []
                if num_selected == 0 or tmp_num == 0
                else random.sample(remainder_keys, num_selected)
            )
            selected_keys += tmp_select
        else:
            u = random.random()
            p = u ** (degree + 1)
            mask = np.random.binomial(1, p, len(remainder_keys))
            selected_keys += [
                remainder_keys[i] for i in range(len(remainder_keys)) if mask[i] == 1
            ]

        selected_indices = [
            self.symptom_name_2_index[self.symptom_data[a]["name"]]
            for a in selected_keys
        ]
        result = np.ones(self.num_symptoms, dtype=int)
        result[selected_indices] = 0

        return result

    def _generate_mask_data_2(self, patho_index, target_symptoms):
        """Generates a mask to obtain a corrupted version of the provided patient data.

        Parameters
        ----------
        patho_index: int
            the index of the considered pathology.
        target_symptoms: list
            list of symptoms that are associated with the simulated pathology.

        Returns
        -------
        result: np.array
            the generated mask.

        """
        assert self.max_turns > 0
        patho_key = self.pathology_index_2_key[patho_index]

        all_binary_symp_keys = [
            a
            for a in self.pathology_data[patho_key]["symptoms"]
            if self.symptom_data_types[
                self.symptom_name_2_index[self.symptom_data[a]["name"]]
            ]
            == "B"
        ]

        all_positive_keys = []
        for val in target_symptoms:
            root_sympt_name, symp_val = self.get_symptom_and_value(val)
            symptom_index = self.symptom_name_2_index[root_sympt_name]
            symptom_key = self.symptom_index_2_key[symptom_index]
            if symp_val is None:  # binary
                all_positive_keys.append(symptom_key)
            else:
                default_value = self.symptom_default_value_mapping[symptom_index]
                # count only if not default value
                if not (str(symp_val) == str(default_value)):
                    all_positive_keys.append(symptom_key)

        curr_positive_evidence = set(all_positive_keys)

        first = random.sample(
            list(curr_positive_evidence & set(all_binary_symp_keys)), 1
        )[0]
        curr_positive_evidence.remove(first)
        curr_positive_evidence_lst = list(curr_positive_evidence)

        max_evidences = min(self.max_turns, len(curr_positive_evidence))
        sample_pos_num_sym = (
            0 if max_evidences <= 0 else random.sample(range(max_evidences + 1), 1)[0]
        )
        sampled_pos_sym = (
            []
            if sample_pos_num_sym <= 0
            else random.sample(curr_positive_evidence_lst, sample_pos_num_sym)
        )
        num_neg_sym = self.max_turns - len(sampled_pos_sym)
        curr_negative_evidence = set(self.symptom_data.keys()) - curr_positive_evidence
        curr_negative_evidence.remove(first)
        curr_negative_evidence_lst = list(curr_negative_evidence)
        max_neg_evidences = min(num_neg_sym, len(curr_negative_evidence_lst))
        sample_neg_num_sym = (
            0 if num_neg_sym <= 0 else random.sample(range(max_neg_evidences), 1)[0]
        )
        sampled_neg_sym = (
            []
            if sample_neg_num_sym <= 0
            else random.sample(curr_negative_evidence_lst, sample_neg_num_sym)
        )
        selected_keys = sampled_pos_sym + sampled_neg_sym + [first]

        selected_indices = [
            self.symptom_name_2_index[self.symptom_data[a]["name"]]
            for a in selected_keys
        ]
        result = np.ones(self.num_symptoms, dtype=int)
        result[selected_indices] = 0

        return result

    def get_data_at_index(self, index, is_corrupted=False, mask=None):
        """Get the eventually corrupted data at the provided index.

        Parameters
        ----------
        index: int
            the index of the patient in the DB to be collected.
        is_corrupted: boolean
            flag indicating whether or not to corrupt the data to be retrieved.
            Default: None
        mask: np.array
            the mask to be used if `is_corrupted` is True. if not specified and
            `is_corrupted` is True, then it will be generated. Default: None

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient. This corresponds
            to his/her age, sex, race, ethnic as well as a randomly
            selected symptom.

        """

        rb_data = self.rb.take(pa.array([index])).to_pydict()
        [
            age,
            race,
            sex,
            ethnic,
            _,
            pathology_index,
            _,
            symptoms,
            diff_indices,
            diff_probas,
            _,
        ] = self._read_rb_data(rb_data)

        # the mask, if any, should cover the whole set of symptoms
        mask = (
            None
            if not is_corrupted
            else (
                (
                    self._generate_mask_data_2(pathology_index, symptoms)
                    if self.max_turns
                    else self._generate_mask_data(pathology_index, symptoms, degree=1)
                )
                if mask is None
                else mask
            )
        )
        assert (mask is None) or (len(mask) >= self.num_symptoms)

        # create the frame
        frame = np.ones((self.num_features,), dtype=self.obs_dtype) * NONE_VAL

        # set the demographic features
        frame = self._set_demo_features(frame, age, sex, race, ethnic)

        num_elts = 0
        for symptom_index in range(self.num_symptoms):
            if (mask is not None) and mask[symptom_index] == 1:
                continue
            num_elts += 1
            start_index = self.symptom_to_obs_mapping[symptom_index][0]
            end_index = self.symptom_to_obs_mapping[symptom_index][1]

            frame, _ = self._apply_symptom_to_observation(
                frame, symptoms, symptom_index, start_index, end_index,
            )

        if self.include_turns_in_state:
            # normalize number of turns
            frame[0] = min(num_elts, self.max_turns) / self.max_turns

        differential_indices = np.array([]) if diff_indices is None else diff_indices
        differential_probas = np.array([]) if diff_probas is None else diff_probas

        # return result
        return frame, (pathology_index, differential_indices, differential_probas)

    def _reset_from_rb_data(self, rb_data, first_symptom=None):
        """Reset the simulator using the provided data.

        This method use the provided first symptom if it is within the
        list of experienced symptoms otherwise the first symptom is chosen
        randomly.

        Parameters
        ----------
        rb_data: dict
            The data used to initialize the simulator.
        first_symptom: str
            The first symptom to be used during initialization. Default: None

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient. This corresponds
            to his/her age, sex, race, ethnic as well as a randomly
            selected symptom.

        """

        self.turns = 0
        self.ordered_actions = []
        self.ordered_rewards = []
        self.frame = np.ones((self.num_features,), dtype=self.obs_dtype) * NONE_VAL

        # read the data from pyarrow record
        [
            self.age,
            self.race,
            self.sex,
            self.ethnic,
            self.target_pathology,
            self.target_pathology_index,
            self.target_pathology_severity,
            self.target_symptoms,
            self.target_differential_indices,
            self.target_differential_probas,
            self.target_initial_symptom,
        ] = self._read_rb_data(rb_data)

        self.inferred_pathology = None

        # set the demographic features
        self.frame = self._set_demo_features(
            self.frame, self.age, self.sex, self.race, self.ethnic
        )

        if self.include_turns_in_state:
            # normalize number of turns
            self.frame[0] = self.turns / self.max_turns

        # set the first symptom chosen randomly
        self.target_symptoms_frame = np.zeros(
            (self.num_features,), dtype=self.obs_dtype
        )
        for f_i in range(self.num_demo_features):
            self.target_symptoms_frame[f_i] = self.frame[f_i]

        # determines the number of relevant symptoms/antecedents for this patient
        [
            self.target_num_symptoms,
            self.target_num_antecedents,
        ] = self._get_relevant_number_of_symptoms_and_antecedent_from_patient(
            self.target_pathology_index,
            self.target_symptoms,
            self.is_reward_relevancy_patient_specific,
        )

        # determine the number of experienced evidences
        self.target_num_experienced_symptoms, self.target_num_experienced_atcds = (
            (self.target_num_symptoms, self.target_num_antecedents)
            if self.is_reward_relevancy_patient_specific
            else self._get_relevant_number_of_symptoms_and_antecedent_from_patient(
                self.target_pathology_index, self.target_symptoms, True
            )
        )
        self.target_num_simulated_evidences = (
            self.target_num_experienced_symptoms + self.target_num_experienced_atcds
        )

        binary_symptoms = []
        considered_symptoms = set(self.target_symptoms + self.all_symptom_names)
        for symptom_name in considered_symptoms:
            root_sympt_name, symp_val = self.get_symptom_and_value(symptom_name)
            symptom_index = self.symptom_name_2_index[root_sympt_name]
            symptom_key = self.symptom_index_2_key[symptom_index]
            # is it part of target
            is_present = symptom_name in self.target_symptoms
            data_type = self.symptom_data_types[symptom_index]
            # first symptom should not be an antecedent
            is_antecedent = self.symptom_data[symptom_key].get("is_antecedent", False)
            if data_type == "B" and is_present and (not is_antecedent):
                binary_symptoms.append(symptom_name)

            # use default value if not present
            if data_type != "B" and not is_present:
                symp_val = self.symptom_default_value_mapping.get(symptom_index)

            f_i = self._from_symptom_index_to_frame_index(symptom_index, symp_val)

            if data_type == "B":
                self.target_symptoms_frame[f_i] = 1 if is_present else 0
            elif data_type == "M":
                self.target_symptoms_frame[f_i] = 1
            else:
                # data_type == "C"
                if not (symptom_index in self.categorical_integer_symptoms):
                    self.target_symptoms_frame[f_i] = 1
                else:
                    val_index = self.symptom_possible_val_mapping[symptom_index][
                        symp_val
                    ]
                    # rescale to 1
                    num = len(self.symptom_possible_val_mapping[symptom_index]) - 1
                    self.target_symptoms_frame[f_i] = val_index / num if num > 0 else 1

        # reset binary symptoms if initial symptom is from the dataset
        binary_symptoms = (
            [self.target_initial_symptom]
            if self.target_initial_symptom is not None and self.use_initial_symptom_flag
            else binary_symptoms
        )

        # only select as first indicator binary symptoms
        assert len(binary_symptoms) > 0, self.target_symptoms + [
            self.current_index,
            self.target_pathology,
        ]
        first_symptom = (
            binary_symptoms[int(len(binary_symptoms) * self.np_random.random())]
            if (first_symptom is None) or (first_symptom not in binary_symptoms)
            else first_symptom
        )
        index_first_symptom = self.symptom_name_2_index[first_symptom]
        first_action = self._from_symptom_index_to_inquiry_action(index_first_symptom)
        frame_index, _ = self._from_inquiry_action_to_frame_index(first_action)
        self.frame[frame_index] = PRES_VAL

        # find the geographic region if any
        self.encoded_geo = None
        if (self.travel_evidence is not None):
            self.encoded_geo = self.encode_geographic_region_if_any(
                self.target_symptoms, self.travel_evidence
            )

        self.is_symptom_updated = True
        self.is_evidence_absent = False
        self.is_symptom_relevant = True
        self.is_symptom_antecedent = False
        self.num_retrieved_evidences = 1
        self.num_retrieved_symptoms = 1
        self.num_retrieved_atcds = 0
        self.ordered_actions.append(first_action)
        self.implicit_symptoms = set()
        self.explicit_inquired_implicit_symptoms = set()

        # return next observation
        return self._next_observation()

    def encode_geographic_region_if_any(self, target_symptoms, geo_sympt_name):
        """Encode the geographic region if any.

        Parameters
        ----------
        target_symptoms: list
            list of the patient evidences.

        Returns
        -------
        code: int
            The encoded info.

        """
        is_geo_present = geo_sympt_name in self.all_symptom_names
        encoded_regio = None
        if is_geo_present:
            indicator = geo_sympt_name + "_@_"
            values = [a for a in target_symptoms if a.startswith(indicator)]
            geo_value = (
                self.travel_negative_response
                if len(values) == 0
                else self.get_symptom_and_value(values[0])[1]
            )
            location = (
                self.default_location
                if geo_value == self.travel_negative_response
                else geo_value
            )
            base_idx = self.symptom_name_2_index.get(geo_sympt_name, -1)

            assert base_idx != -1, (
                f"The travel code {geo_sympt_name} is not defined. "
                f"please configure the simulator accordingly."
            )

            assert self.symptom_possible_val_mapping.get(base_idx)
            assert location in self.symptom_possible_val_mapping[base_idx]
            encoded_regio = self.symptom_possible_val_mapping[base_idx][location]
        return encoded_regio

    def reset_with_index(self, index):
        """Reset an interaction between a specific patient and an automated agent.

        Parameters
        ----------
        index: int
            the index of the patient of interest.

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient. This corresponds
            to his/her age, sex, race, ethnic as well as a randomly
            selected symptom.

        """
        assert index >= 0 and index < self.rb.num_rows
        self.current_index = index
        rb_data = self.rb.take(pa.array([self.current_index])).to_pydict()
        return self._reset_from_rb_data(rb_data)

    def reset(self):
        """Reset an interaction between a patient and an automated agent.

        Parameters
        ----------

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient. This corresponds
            to his/her age, sex, race, ethnic as well as a randomly
            selected symptom.

        """

        self.current_index = int(self.rb.num_rows * self.np_random.random())
        rb_data = self.rb.take(pa.array([self.current_index])).to_pydict()
        return self._reset_from_rb_data(rb_data)

    def get_num_indices(self):
        """Return the number of patient indices within the environment.

        Parameters
        ----------

        Returns
        -------
        number: int
            The number of patient indices.

        """
        return self.rb.num_rows

    def _next_observation(self):
        """Get the next observation from the simulator.

        Parameters
        ----------

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient.

        """
        if self.frame is None:
            return np.ones((self.num_features,), dtype=self.obs_dtype) * NONE_VAL

        if self.include_turns_in_state:
            # normalize number of turns
            self.frame[0] = self.turns / self.max_turns
        return np.array(self.frame)

    def _is_an_inquiry_action(self, action):
        """Check if an action is an inquiry action.

        Parameters
        ----------
        action: int
            the action to be checked.

        Returns
        -------
        result: bool
            True if the action is an inqury, False otherwise.

        """
        if self._has_int_action():
            return action < len(self.symptom_index_2_key)
        else:
            return action[0] == 0

    def _is_a_diagnosis_action(self, action):
        """Check if an action is a diagnosis action.

        Parameters
        ----------
        action: int
            the action to be checked.

        Returns
        -------
        result: bool
            True if the action is a diagnosis, False otherwise.
        """
        if self._has_int_action():
            return action >= len(self.symptom_index_2_key)
        else:
            return action[0] == 1

    def _from_inquiry_action_to_symptom_index(self, action):
        """Get the index of the symptom corresponding to the inquiry action.

        Parameters
        ----------
        action: int
            the action from which we want to find the corresponding
            symptom index.

        Returns
        -------
        index: int
            index of the symptom associated with the provided action.

        """
        if self._has_int_action():
            symptom_index = action
        else:
            symptom_index = action[1]

        if isinstance(symptom_index, np.ndarray):
            symptom_index = symptom_index.item()

        return symptom_index

    def _from_symptom_index_to_inquiry_action(self, symptom_index):
        """Get the inquiry action corresponding to the provided symptom index.

        Parameters
        ----------
        symptom_index: int
            the index of the symptom from which we want to find
            the corresponding action.

        Returns
        -------
        action: int
            the inquiry action associated with the provided symptom.

        """
        if self._has_int_action():
            return symptom_index
        else:
            return [0, symptom_index, 0]

    def _from_symptom_index_to_frame_index(self, symptom_index, symptom_val=None):
        """Get the frame index corresponding to the symptom_index.

        Parameters
        ----------
        symptom_index: int
            the index of the symptom from which we want to find
            the corresponding action.
        symptom_val: obj
            the value associated to the the symptom indexed by
            the provided index. Default: None

        Returns
        -------
        index: int
            frame index associated with the provided action.

        """
        data_type = self.symptom_data_types[symptom_index]

        if data_type == "B":
            return self.symptom_to_obs_mapping[symptom_index][0]
        elif data_type == "C" and (symptom_index in self.categorical_integer_symptoms):
            return self.symptom_to_obs_mapping[symptom_index][0]
        else:
            if symptom_val is None:
                raise ValueError("The expected value is NOT supposed to be None.")
            idx = self.symptom_possible_val_mapping[symptom_index][symptom_val]
            return self.symptom_to_obs_mapping[symptom_index][0] + idx

    def _from_inquiry_action_to_frame_index(self, action):
        """Get the frame index corresponding to the inquiry action.

        Parameters
        ----------
        action: int
            the action from which we want to find the corresponding
            frame index.

        Returns
        -------
        indices: couple of int
            frame indices interval [start, end] associated with
            the provided action.

        """
        symptom_index = self._from_inquiry_action_to_symptom_index(action)

        start_idx = self.symptom_to_obs_mapping[symptom_index][0]
        end_idx = self.symptom_to_obs_mapping[symptom_index][1]

        return start_idx, end_idx

    def _from_diagnosis_action_to_patho_index(self, action):
        """Get the index of the pathology corresponding to the diagnosis action.

        Parameters
        ----------
        action: int
            the action from which we want to find the corresponding
            pathology index.

        Returns
        -------
        index: int
            index of the pathology associated with the provided action.

        """
        if self._has_int_action():
            return action - len(self.symptom_index_2_key)
        else:
            return action[2]

    def check_action_validity(self, action):
        """Check if a gven action is valid.

        Parameters
        ----------
        action: int
            the action to be checked.

        Returns
        -------
        result: bool
            True if it is valid otherwise, raised an exception.

        """
        if self._has_int_action():
            if action >= self.num_actions[0]:
                raise ValueError(
                    "Invalid action! must be in interval [{}, {}]: {}".format(
                        0, self.num_actions[0], action
                    )
                )
        else:
            assert len(action) == 3, "action len must be 3"
            for i in range(3):
                if action[i] >= self.num_actions[i]:
                    raise ValueError(
                        f"Invalid action! index [{i}] - must be in interval"
                        f" [0, {self.num_actions[i]}]: {action[i]}"
                    )

        return True

    def _set_symptom_to_observation(
        self,
        frame,
        all_symptoms,
        symptom_index,
        start_index,
        end_index,
        check_consistency=True,
    ):
        """Utility to write out the value of an inquired symptom into the state data.

        For boolean symptoms or integer-based categorical symptoms, we just set the
        corresponding entry in the state space with the proper value.

        Otherwise (multi-choice and generic categorical symptoms), we first set all
        the index range to ABS_VAL before setting to PRES_VAL all the indices associated
        to the actual values related to the symptom from the simulated patient.

        This function differs from `_apply_symptom_to_observation` in that it allows to
        implicitly set linked symptom values when needed.

        Parameters
        ----------
        frame: np.array
            the state data to be updated.
        all_symptoms: list
            list of experienced symptoms to be considered.
        symptom_index: int
            the index of the inquired symptom.
        start_index: int
            the start index of the portion of the observation state dedicated
            to the provided symptom.
        end_index: int
            the end index of the portion of the observation state dedicated
            to the provided symptom.
        check_consistency: boolean
            if True, then a consistency check is made with respect to the provided
            `all_symptoms` when implicitly set the linked symptoms. Default: True

        Returns
        -------
        frame: np.array
            the updated state data.
        no_evidence_flag: boolean
            True if the provided symptom_index is not part of the `all_symptoms`
            or it is set to the default value.
        implicit_symptoms: set
            Set of symptoms that are implicitly set in the frame data.

        """
        frame, no_evidence_flag = self._apply_symptom_to_observation(
            frame, all_symptoms, symptom_index, start_index, end_index
        )
        implicit_symptoms = None

        # here, the base question has been asked and the answer is no
        # we need to implicitly set all the linked questions to false
        # assuming they were not asked before
        if (symptom_index in self.all_linked_symptom_indices) and no_evidence_flag:
            implicit_symptoms = set()
            for linked_symp_idx in self.all_linked_symptom_indices[symptom_index]:
                # get the associated frame index
                tmp_start_idx = self.symptom_to_obs_mapping[linked_symp_idx][0]
                tmp_end_idx = self.symptom_to_obs_mapping[linked_symp_idx][1]

                # check if not already asked
                if np.all(frame[tmp_start_idx:tmp_end_idx] == NONE_VAL):
                    # use the provided all_symptoms only if check_consistency is needed
                    tmp_symptoms = all_symptoms if check_consistency else []
                    frame, tmp_no_evidence_flag = self._apply_symptom_to_observation(
                        frame, tmp_symptoms, linked_symp_idx, tmp_start_idx, tmp_end_idx
                    )
                    # values for linked_sympt_idx shouldn't be present in all_symptoms
                    assert tmp_no_evidence_flag
                    # save the implicit symptoms
                    implicit_symptoms.add(linked_symp_idx)

        # here, a linked question is asked and the answer is yes
        # we need to set the base question to True assuming it has not yet been inquired
        if (
            symptom_index in self.all_linked_reverse_symptom_indices
        ) and not no_evidence_flag:
            implicit_symptoms = (
                set() if implicit_symptoms is None else implicit_symptoms
            )
            # get the base_idx
            base_symp_idx = self.all_linked_reverse_symptom_indices[symptom_index]
            # get the associated frame index
            tmp_start_idx = self.symptom_to_obs_mapping[base_symp_idx][0]
            tmp_end_idx = self.symptom_to_obs_mapping[base_symp_idx][1]

            # check if not already asked
            if np.all(frame[tmp_start_idx:tmp_end_idx] == NONE_VAL):
                # use the provided all_symptoms only if check_consistency is needed
                tmp_symptoms = (
                    all_symptoms
                    if check_consistency
                    else [
                        self.symptom_data[self.symptom_index_2_key[base_symp_idx]][
                            "name"
                        ]
                    ]
                )
                frame, tmp_no_evidence_flag = self._apply_symptom_to_observation(
                    frame, tmp_symptoms, base_symp_idx, tmp_start_idx, tmp_end_idx
                )
                # values for base_symp_idx should be present in all_symptoms
                assert not tmp_no_evidence_flag
                # save the implicit symptoms
                implicit_symptoms.add(base_symp_idx)

        # return value
        return frame, no_evidence_flag, implicit_symptoms

    def _apply_symptom_to_observation(
        self, frame, all_symptoms, symptom_index, start_index, end_index
    ):
        """Utility to write out the value of an inquired symptom into the state data.

        For boolean symptoms or integer-based categorical symptoms, we just set the
        corresponding entry in the state space with the proper value.

        Otherwise (multi-choice and generic categorical symptoms), we first set all
        the index range to ABS_VAL before setting to PRES_VAL all the indices associated
        to the actual values related to the symptom from the simulated patient.

        Parameters
        ----------
        frame: np.array
            the state data to be updated.
        all_symptoms: list
            list of experienced symptoms to be considered.
        symptom_index: int
            the index of the inquired symptom.
        start_index: int
            the start index of the portion of the observation state dedicated
            to the provided symptom.
        end_index: int
            the end index of the portion of the observation state dedicated
            to the provided symptom.

        Returns
        -------
        frame: np.array
            the updated state data.
        no_evidence_flag: boolean
            True if the provided symptom_index is not part of the `all_symptoms`
            or it is set to the default value.

        """

        # set the abs value by default for the full frame range
        for idx in range(start_index, end_index):
            frame[idx] = ABS_VAL

        data_type = self.symptom_data_types[symptom_index]
        key_symptom = self.symptom_index_2_key[symptom_index]
        symptom_name = self.symptom_data[key_symptom]["name"]
        indicator = symptom_name + "_@_"
        is_absent_or_default_value = True

        if data_type == "B":
            if symptom_name in all_symptoms:
                frame[start_index] = PRES_VAL
                is_absent_or_default_value = False
        else:
            values = [a for a in all_symptoms if a.startswith(indicator)]
            default_value = indicator + str(
                self.symptom_default_value_mapping[symptom_index]
            )
            if len(values) == 0:
                # use default value
                values.append(default_value)
            if data_type == "C":
                # only one value for categorical data
                assert len(values) == 1, (
                    f"Only 1 value for categorical symptom. Got {len(values)}"
                    f" values for symptom {symptom_name}"
                )
                is_absent_or_default_value = values[0] == default_value
            if data_type == "M":
                is_absent_or_default_value = (len(values) == 1) and (
                    values[0] == default_value
                )
            for v in values:
                _, symp_val = self.get_symptom_and_value(v)
                f_i = self._from_symptom_index_to_frame_index(symptom_index, symp_val)
                assert f_i >= start_index and f_i < end_index
                if symptom_index in self.categorical_integer_symptoms:
                    val_index = self.symptom_possible_val_mapping[symptom_index][
                        symp_val
                    ]
                    # scale value
                    num = len(self.symptom_possible_val_mapping[symptom_index])
                    scaled = NONE_VAL + ((PRES_VAL - NONE_VAL) * (val_index + 1) / num)
                    frame[f_i] = scaled
                else:
                    frame[f_i] = PRES_VAL
        return frame, is_absent_or_default_value

    def _take_action(self, action):
        """Execute an action and get the answer of the patient.

        Parameters
        ----------
        action: int
            the action to be executed.

        Returns
        -------
        None

        """
        self.is_symptom_updated = False
        self.is_evidence_absent = True
        self.is_symptom_relevant = False
        self.is_symptom_antecedent = False
        self.check_action_validity(action)

        # is it an inquiry for a symptom ?
        if self._is_an_inquiry_action(action):
            start_index, end_index = self._from_inquiry_action_to_frame_index(action)
            symptom_index = self._from_inquiry_action_to_symptom_index(action)
            symptom_key = self.symptom_index_2_key[symptom_index]
            is_implicit = (
                symptom_index in self.implicit_symptoms
                and symptom_index not in self.explicit_inquired_implicit_symptoms
            )

            # if a symptom is implicitly set, when receivin an explicit question about
            # it for the first time, we act as if the symptom was not set
            if np.all(self.frame[start_index:end_index] == NONE_VAL) or is_implicit:
                self.is_symptom_updated = True
                if is_implicit:
                    self.explicit_inquired_implicit_symptoms.add(symptom_index)

            self.is_symptom_relevant = self._is_direct_symptom_relevancy(
                self.target_pathology_index, symptom_index, self.target_symptoms
            )

            self.is_symptom_antecedent = self.symptom_data[symptom_key].get(
                "is_antecedent", False
            )

            self.frame, self.is_evidence_absent, imp = self._set_symptom_to_observation(
                self.frame,
                self.target_symptoms,
                symptom_index,
                start_index,
                end_index,
                True,
            )
            if imp is not None:
                self.implicit_symptoms.update(imp)

            # update the count of retrieved evidences
            if self.is_symptom_updated and not self.is_evidence_absent:
                self.num_retrieved_evidences += 1
                if self.is_symptom_antecedent:
                    self.num_retrieved_atcds += 1
                else:
                    self.num_retrieved_symptoms += 1

            # we make sure all the index range have been set
            assert np.all(self.frame[start_index:end_index] != NONE_VAL)

        elif self._is_a_diagnosis_action(action):  # it is a diagnostic
            patho_index = self._from_diagnosis_action_to_patho_index(action)
            key_patho = self.pathology_index_2_key[patho_index]
            patho_name = self.pathology_data[key_patho]["condition_name"]
            self.inferred_pathology = patho_name

        else:
            raise ValueError("Invalid action type: not an Inquiry/Diagnosis !")

        # save the action
        self.ordered_actions.append(action)

    def _is_interaction_done(self, action):
        """Check if an interaction session will be done after executing `action`.

        Parameters
        ----------
        action: int
            the action to be executed.

        Returns
        -------
        done: bool
            True if the interaction is over. False otherwise.

        """
        if (self.max_turns is not None) and (self.max_turns > 0):
            if self.turns >= self.max_turns:
                return True

        # if a diagnosis is made
        if self._is_a_diagnosis_action(action):
            return True

        # here, we are left with inquiry question
        if self.stop_if_repeated_question:
            if not self.is_symptom_updated:
                return True

        return False

    def _get_relevant_number_of_symptoms_and_antecedent_from_patient(
        self, patho_index, patient_data, patient_specific_relevancy_flag
    ):
        """Get the number of symptoms/antecedents within the provided patient data.

        Parameters
        ----------
        patho_index: int
            the index of the considered pathology.
        patient_data: list of str
            the list of symptoms experienced by the considered patient.
        patient_specific_relevancy_flag: bool
            flag indicating whether or not to be pathology or patient specific.

        Returns
        -------
        num_symptoms: int
            number of relevant symptoms found.
        num_antecedents: int
            number of relevant antecedents found.

        """
        patho_key = self.pathology_index_2_key[patho_index]
        if not patient_specific_relevancy_flag:
            num_symptoms = len(self.pathology_data[patho_key]["symptoms"])
            num_antecedents = len(self.pathology_data[patho_key].get("antecedents", {}))
        else:
            num_symptoms = 0
            num_antecedents = 0

            # keep the value for different symptoms
            data_dict = {}
            for symptom_name in patient_data:
                root_sympt_name, symp_val = self.get_symptom_and_value(symptom_name)
                symptom_index = self.symptom_name_2_index[root_sympt_name]
                if symptom_index not in data_dict:
                    data_dict[symptom_index] = []
                data_dict[symptom_index].append(symp_val)

            for symptom_index in data_dict.keys():
                data_type = self.symptom_data_types[symptom_index]
                symptom_key = self.symptom_index_2_key[symptom_index]
                is_antecedent = self.symptom_data[symptom_key].get(
                    "is_antecedent", False
                )

                # if we have a binary symptom or symptom with multiple values
                if data_type == "B" or len(data_dict[symptom_index]) > 1:
                    if is_antecedent:
                        num_antecedents += 1
                    else:
                        num_symptoms += 1
                else:
                    assert len(data_dict[symptom_index]) == 1
                    default_value = self.symptom_default_value_mapping[symptom_index]
                    # count only if not default value
                    if not (str(data_dict[symptom_index][0]) == str(default_value)):
                        if is_antecedent:
                            num_antecedents += 1
                        else:
                            num_symptoms += 1

        return num_symptoms, num_antecedents

    def _is_direct_symptom_relevancy(self, patho_index, symptom_index, target_symptoms):
        """Check if a symptom is part of the symptom set describing a given pathology.

        Parameters
        ----------
        patho_index: int
            the index of the considered pathology.
        symptom_index: int
            the index of the considered symptom.
        target_symptoms: list
            list of symptoms that are associated with the simulated pathology.

        Returns
        -------
        result: bool
            True if the symptom is part of the symptom set, False otherwise.

        """
        patho_key = self.pathology_index_2_key[patho_index]
        symptom_key = self.symptom_index_2_key[symptom_index]

        if not self.is_reward_relevancy_patient_specific:
            # check if symptom is part of the description of the pathology
            return symptom_key in self.pathology_data[patho_key][
                "symptoms"
            ] or symptom_key in self.pathology_data[patho_key].get("antecedents", {})
        else:
            data_type = self.symptom_data_types[symptom_index]

            # is it part of target
            symptom_name = self.symptom_data[symptom_key]["name"]
            is_present = symptom_name in target_symptoms

            if data_type == "B":
                return is_present

            indicator = symptom_name + "_@_"
            values = [a for a in target_symptoms if a.startswith(indicator)]

            # no data can be found in target that are related to the symptoms
            if len(values) == 0:
                return False

            # if it is a categorical symptom (C), then values length must be 1
            assert (len(values) == 1) or (data_type != "C")

            # more than one input provided, meaning it is present (not default value)
            if len(values) > 1:
                return True

            # one input provided (check if not default value)
            _, symptom_value = self.get_symptom_and_value(values[0])
            default_value = self.symptom_default_value_mapping[symptom_index]
            is_default_value = str(symptom_value) == str(default_value)

            # True if not default value, False otherwise
            return not is_default_value

    def _compute_generic_reward(
        self,
        max_turns,
        turns,
        action,
        target_pathology_index,
        target_differential_indices,
        target_differential_probas,
        target_symptoms,
        is_symptom_updated,
        stop_if_repeated_question,
        reward_config,
    ):
        """Utility to compute the reward score.

        It computes the reward associated to the taken `action`
        given the context and the reward configuration.

        Parameters
        ----------
        max_turns: int, None
            the maximum number of allowed turns.
        turns: int
            the current turn number.
        action: int
            the action to be executed.
        target_pathology_index: int
            the index of the pathology being simulated.
        target_differential_indices: np.array
            the pathology indices associated with the patient differential.
        target_differential_probas: np.array
            the probability associated with the pathologies that are part
            of the patient differential.
        target_symptoms: list
            list of symptoms that are associated with the simulated pathology.
        is_symptom_updated: bool
            False is the provided action (symptom) is a repeated action in the
            interaction session. otherwise, it is True.
        stop_if_repeated_question: bool
            indicator to indicate if the interactive session should end up
            if a symptom is inquired more than once.
        reward_config: the reward configuration.

        Returns
        -------
        reward: float
            the reward.

        """

        # symptom inquiry
        if self._is_an_inquiry_action(action):
            reward = 0
            if not (max_turns is None) and (turns >= max_turns):
                reward = (
                    reward_config.reward_on_reaching_max_turns
                    + self._get_sympt_atcd_missing_recall_reward()
                )
            if stop_if_repeated_question and not is_symptom_updated:
                return reward + reward_config.reward_on_repeated_action
            else:
                if callable(reward_config.reward_on_intermediary_turns):
                    reward += reward_config.reward_on_intermediary_turns(turns)
                else:
                    reward += reward_config.reward_on_intermediary_turns
                if not is_symptom_updated:
                    reward += reward_config.reward_on_repeated_action
                else:
                    # check for relevancy:
                    symptom_index = self._from_inquiry_action_to_symptom_index(action)
                    if self._is_direct_symptom_relevancy(
                        target_pathology_index, symptom_index, target_symptoms
                    ):
                        reward += reward_config.reward_on_relevant_symptom_inquiry
                    else:
                        reward += reward_config.reward_on_irrelevant_symptom_inquiry
                return reward
        elif self._is_a_diagnosis_action(action):  # diagnosis
            patho_index = self._from_diagnosis_action_to_patho_index(action)
            has_diff = (target_differential_indices is not None) and (
                target_differential_probas is not None
            )
            sympt_ant_missing_recall_reward = (
                self._get_sympt_atcd_missing_recall_reward()
            )
            if (not has_diff) and (patho_index == target_pathology_index):  # match
                return (
                    reward_config.reward_on_correct_diagnosis
                    + sympt_ant_missing_recall_reward
                )
            elif has_diff and np.any(patho_index == target_differential_indices):
                pos = np.where(patho_index == target_differential_indices)[0][0]
                # get the relative weight with respect to the top-1 pathology
                tmp_fl = reward_config.rescale_correct_diagnosis_reward_on_differential
                coef = (
                    1
                    if not tmp_fl
                    else target_differential_probas[pos] / target_differential_probas[0]
                )
                return (
                    coef * reward_config.reward_on_correct_diagnosis
                ) + sympt_ant_missing_recall_reward
            else:
                return (
                    reward_config.reward_on_missing_diagnosis
                ) + sympt_ant_missing_recall_reward
        else:
            raise ValueError("Invalid action type: not an Inquiry/Diagnosis !")

    def _get_sympt_atcd_missing_recall_reward(self):
        """Returns the reward for the missed symptoms/antecendents by the agent.

        Parameters
        ----------
        None

        Returns
        -------
        reward: float
            Reward for the missed symptoms/antecendents by the agent.

        """
        missing_inquired_evidences = (
            self.target_num_simulated_evidences - self.num_retrieved_evidences
        )
        missing_inquired_symptoms = (
            self.target_num_experienced_symptoms - self.num_retrieved_symptoms
        )
        missing_inquired_atcds = (
            self.target_num_experienced_atcds - self.num_retrieved_atcds
        )
        missing_total = missing_inquired_symptoms + missing_inquired_atcds
        assert missing_inquired_evidences >= 0
        assert missing_inquired_symptoms >= 0
        assert missing_inquired_atcds >= 0
        assert missing_inquired_evidences == missing_total

        reward = 0
        reward += (
            missing_inquired_symptoms
            * self.reward_config.reward_on_missing_symptoms_recall
        )
        reward += (
            missing_inquired_atcds * self.reward_config.reward_on_missing_atcd_recall
        )
        return reward

    def _compute_reward(self, action):
        """Function to compute the reward of an `action` at the current state.

        Parameters
        ----------
        action: int
            the action to be executed.

        Returns
        -------
        reward: float
            the reward.

        """
        reward = self._compute_generic_reward(
            self.max_turns,
            self.turns,
            action,
            self.target_pathology_index,
            self.target_differential_indices,
            self.target_differential_probas,
            self.target_symptoms,
            self.is_symptom_updated,
            self.stop_if_repeated_question,
            self.reward_config,
        )
        # save the reward
        self.ordered_rewards.append(reward)
        return reward

    def step(self, action):
        """Execute an action.

        Parameters
        ----------
        action: int
            the action to be executed.

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient. This corresponds
            to his/her age, sex, race, ethnic as well as a randomly
            selected symptom.
        reward: float
            The reward associated to the provided action at the
            current state.
        done: bool
            a flag indicating wether the interaction session ends or not.
        info: dict
            additional info.

        """

        self._take_action(action)
        self.turns += 1

        done = self._is_interaction_done(action)
        reward = self._compute_reward(action)

        obs = self._next_observation()

        env_info = {
            "sim_patient": self.target_symptoms_frame,
            "sim_patho": self.target_pathology_index,
            "sim_severity": self.target_pathology_severity,
            "sim_evidence": 1 if not self.is_evidence_absent else 0,
            "sim_differential_indices": self.target_differential_indices,
            "sim_differential_probas": self.target_differential_probas,
            "sim_age": self.age,
            "sim_sex": encode_sex(self.sex),
            "sim_race": encode_race(self.race),
            "sim_timestep": self.turns,
            "sim_total_num_pathos": self.num_pathos,
            "is_repeated_action": 1 if not self.is_symptom_updated else 0,
            "is_relevant_action": 1 if self.is_symptom_relevant else 0,
            "is_antecedent_action": 1 if self.is_symptom_antecedent else 0,
            "sim_num_symptoms": self.target_num_symptoms,
            "sim_num_antecedents": self.target_num_antecedents,
            "sim_num_evidences": self.target_num_simulated_evidences,
            "first_symptom": self._from_inquiry_action_to_symptom_index(
                self.ordered_actions[0]
            ),
            "current_symptom": self._from_inquiry_action_to_symptom_index(action)
            if self._is_an_inquiry_action(action)
            else -1,
            "diagnostic": self.pathology_name_2_index[self.inferred_pathology]
            if (self.inferred_pathology is not None)
            else -1,
        }
        if self.encoded_geo is not None:
            env_info["sim_geo"] = self.encoded_geo

        return obs, reward, done, env_info

    def _render_header(self, action_index, file):
        """Write out the demo info and the first symptom of the current session.

        Parameters
        ----------
        action_index: int
            the index of the first symptom provided.
        file: filestream
            stream where to log the data.

        Returns
        -------
        None

        """
        tmp_init_idx = 1 if self.include_turns_in_state else 0
        race_code = (
            f"{self.frame[tmp_init_idx + 2]}"
            if self.include_race_in_state
            else "not included"
        )
        diff = 0 if self.include_race_in_state else 1
        ethnic_code = (
            f"{self.frame[tmp_init_idx + 3 - diff]}"
            if self.include_ethnicity_in_state
            else "not included"
        )
        file.write("------------------------------------\n")
        file.write(f"Age: {self.age} ({self.frame[tmp_init_idx + 0]})\n")
        file.write(f"Sex: {self.sex} ({self.frame[tmp_init_idx + 1]})\n")
        file.write(f"Race: {self.race} ({race_code})\n")
        file.write(f"Ethnic: {self.ethnic} ({ethnic_code})\n")
        file.write("SIMULATED PATHOLOGY:\n")
        file.write(f"\t{self.target_pathology}\n")
        if (
            self.target_differential_indices is not None
            and self.target_differential_probas is not None
        ):
            file.write("DXA Differential Diagnosis:\n")
            target_diff_arr = zip(
                self.target_differential_indices, self.target_differential_probas
            )
            dxa_diff_diag_str = convert_patho_idx_to_patho_str(
                target_diff_arr, self.pathology_index_2_key, "\t", "\n"
            )
            file.write(f"{dxa_diff_diag_str}")
        file.write("REAL SYMPTOMS:\n")
        for real_symptom in sorted(self.target_symptoms):
            file.write(f"\t-{real_symptom}\n")
        file.write("SYMPTOM INQUIRY: \n")
        # the first symptom is always a boolean symptom
        start_index, end_index = self._from_inquiry_action_to_frame_index(action_index)
        symptom_index = self._from_inquiry_action_to_symptom_index(action_index)
        self._render_symptom(symptom_index, start_index, end_index, file)

    def _render_step(
        self, action_index, turn_index, file, patho_predictions=None, num=5
    ):
        """Write out the demo info at a given step of the current session.

        Parameters
        ----------
        action_index: int
            the index of action undertook.
        turn_index: int
            the index of turn.
        file: filestream
            stream where to log the data.
        patho_predictions: np.array
            the prediction regarding the pathologies: Default: None
        num: int
            the number of top pathologies to ouput: Default: 5

        Returns
        -------
        None

        """
        is_over = False
        if self._is_an_inquiry_action(action_index):
            symptom_index = self._from_inquiry_action_to_symptom_index(action_index)
            start_index, end_index = self._from_inquiry_action_to_frame_index(
                action_index
            )
            self._render_symptom(symptom_index, start_index, end_index, file)
            max_turn_on = (self.max_turns is not None) and (self.max_turns > 0)
            if max_turn_on and (turn_index >= self.max_turns):
                file.write("DIAGNOSIS: \n")
                file.write("\tMaximum of turns reached\n")
                is_over = True
        elif self._is_a_diagnosis_action(action_index):  # diagnosis
            file.write("DIAGNOSIS: \n")
            patho_index = self._from_diagnosis_action_to_patho_index(action_index)
            key_patho = self.pathology_index_2_key[patho_index]
            patho_name = self.pathology_data[key_patho]["condition_name"]
            file.write(f"\t{patho_name}\n")
            is_over = True
        else:
            raise ValueError("Invalid action type: not an Inquiry/Diagnosis !")

        if is_over:
            file.write("REWARD: \n")
            file.write(f"\t{sum(self.ordered_rewards)}\n")

            if patho_predictions is not None:
                file.write("DIFFERENTIAL DIAGNOSIS: \n")
                assert len(patho_predictions.shape) <= 2
                if len(patho_predictions.shape) == 2:
                    assert patho_predictions.shape[0] == 1
                    patho_predictions = patho_predictions[0]
                topk = np.argsort(patho_predictions, axis=0)[-num:]
                for i in range(len(topk), 0, -1):
                    patho_index = topk[i - 1]
                    key_patho = self.pathology_index_2_key[patho_index]
                    patho_name = self.pathology_data[key_patho]["condition_name"]
                    file.write(f"\t{patho_name}: {patho_predictions[patho_index]}\n")

    def _render_symptom(self, symptom_index, start_index, end_index, file):
        """Write out the inquired symptom info a given step of the current session.

        Parameters
        ----------
        symptom_index: int
            the index of the inquired symptom.
        start_index: int
            the start index of the portion of the observation state dedicated
            to the provided symptom.
        end_index: int
            the end index of the portion of the observation state dedicated
            to the provided symptom.
        file: filestream
            stream where to log the data.

        Returns
        -------
        None

        """
        key_symptom = self.symptom_index_2_key[symptom_index]
        symptom_name = self.symptom_data[key_symptom]["name"]
        data_type = self.symptom_data_types[symptom_index]
        if data_type == "B":
            symp_answer = self.frame[start_index]
            symp_answer = "Y" if symp_answer == PRES_VAL else "N"
            file.write(f"\t{symptom_name}: {symp_answer}\n")
        else:
            if symptom_index in self.categorical_integer_symptoms:
                num = len(self.symptom_possible_val_mapping[symptom_index])
                scaled = self.frame[start_index]
                val_index = ((scaled - NONE_VAL) * num) / (PRES_VAL - NONE_VAL)
                val_index = int(val_index - 1)
                val = self.symptom_data[key_symptom]["possible-values"][val_index]
                symp_answer = val
                file.write(f"\t{symptom_name}: {symp_answer}\n")
            else:
                file.write(f"\t{symptom_name}:\n")
                found = False
                for f_i in range(start_index, end_index):
                    if self.frame[f_i] == PRES_VAL:
                        idx = f_i - start_index
                        symp_answer = self.symptom_data[key_symptom]["possible-values"][
                            idx
                        ]
                        file.write(f"\t\t{symp_answer}\n")
                        found = True
                if not found:
                    file.write("\t\tNA\n")

    def render(self, mode="step", filename="render.txt", patho_predictions=None, num=5):
        """Render the environment.

        Parameters
        ----------
        mode: str
            'step': render only the information regarding
                    the current stage of the interaction.
            'all':  render all the information collected so far.
        filename: str
            name of the file wher to write those information.
        patho_predictions: np.array
            the prediction regarding the pathologies: Default: None
        num: int
            the number of top pathologies to ouput: Default: 5

        Returns
        -------
        None

        """
        if self.frame is None:
            return
        file = open(filename, "a+")
        if mode == "step":
            action_index = self.ordered_actions[self.turns]
            if self.turns == 0:
                self._render_header(action_index, file)
            else:
                self._render_step(
                    action_index, self.turns, file, patho_predictions, num
                )
        elif mode == "all":
            for i in range(len(self.ordered_actions)):
                action_index = self.ordered_actions[i]
                if i == 0:
                    self._render_header(action_index, file)
                else:
                    self._render_step(action_index, i, file, patho_predictions, num)
        file.close()
