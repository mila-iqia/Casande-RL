import os

import numpy as np
import torch
from gym.envs.registration import register
from rlpyt.envs.gym import make as gym_make

from chloe.utils.algo_components import AugSamplesToBuffer
from chloe.utils.replay_buffer_utils import ReplayBufferFactory
from chloe.utils.replay_components import AugSamplesFromReplay


class TestReplayBuffer(object):
    def examples_to_buffer(self, examples):
        """Defines how to initialize the replay buffer from examples.
        """
        return AugSamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            sim_patho=getattr(examples["env_info"], "sim_patho", None),
            sim_patient=getattr(examples["env_info"], "sim_patient", None),
            sim_severity=getattr(examples["env_info"], "sim_severity", None),
            sim_evidence=getattr(examples["env_info"], "sim_evidence", None),
            sim_timestep=getattr(examples["env_info"], "sim_timestep", None),
            sim_differential_indices=getattr(
                examples["env_info"], "sim_differential_indices", None
            ),
            sim_differential_probas=getattr(
                examples["env_info"], "sim_differential_probas", None
            ),
        )

    def samples_to_buffer(self, examples):
        sim_differential_indices = getattr(
            examples["env_info"], "sim_differential_indices", None
        )
        sim_differential_probas = getattr(
            examples["env_info"], "sim_differential_probas", None
        )
        if sim_differential_indices is not None:
            sim_differential_indices = sim_differential_indices.reshape(1, 1, -1)
        if sim_differential_probas is not None:
            sim_differential_probas = sim_differential_probas.reshape(1, 1, -1)
        return AugSamplesToBuffer(
            observation=examples["observation"].reshape(1, 1, -1),
            action=np.array(examples["action"]).reshape(1, 1, -1),
            reward=examples["reward"].reshape(1, 1, -1),
            done=np.array(examples["done"]).reshape(1, 1, -1),
            sim_patho=np.array(
                getattr(examples["env_info"], "sim_patho", None)
            ).reshape(1, 1, -1),
            sim_patient=np.array(
                getattr(examples["env_info"], "sim_patient", None)
            ).reshape(1, 1, -1),
            sim_severity=np.array(
                getattr(examples["env_info"], "sim_severity", None)
            ).reshape(1, 1, -1),
            sim_evidence=np.array(
                getattr(examples["env_info"], "sim_evidence", None)
            ).reshape(1, 1, -1),
            sim_timestep=np.array(
                getattr(examples["env_info"], "sim_timestep", None)
            ).reshape(1, 1, -1),
            sim_differential_indices=sim_differential_indices,
            sim_differential_probas=sim_differential_probas,
        )

    def get_random_example_from_env(self, env):
        """get a random examples data from the provided env
        """
        a = np.random.randint(0, 45)
        o, r, d, env_info = env.step(a)
        r = np.asarray(r, dtype="float32")

        if d:
            env.reset()

        examples = dict()
        examples["observation"] = o
        examples["reward"] = r
        examples["done"] = d
        examples["env_info"] = env_info
        examples["action"] = a
        examples["agent_info"] = None
        return examples

    def test_replay_buffer(self, tmpdir):

        GYM_ENV_ID = "simPa-v0"

        # try to register the env in gym if not yet done
        try:
            register(
                id=GYM_ENV_ID,
                entry_point="chloe.simulator.simulator:PatientInteractionSimulator",
            )
        except Exception:
            pass

        sample_data_patient = (
            "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN,AGE_END,PATHOLOGY,"
            "NUM_SYMPTOMS,SYMPTOMS\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,"
            "3,3,Acute bronchitis,6,Nasal"
            " congestion:28;Sore throat:29;Wheezing:25;Coryza:47;Fever:36"
            ";Cough:48\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3"
            ",Acute fatty liver of "
            "pregnancy (AFLP),3,Itching of skin:47;Cross-eyed:35;Emotional"
            " symptoms:34\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3,"
            "Acute bronchiolitis,8,"
            "Difficulty breathing:34;Nasal congestion:44;Irritable infant:44"
            ";Vomiting:32;Pulling"
            " at ears:45;Wheezing:26;Fever:27;Cough:42\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3,Acute"
            " otitis media,8,"
            "Nasal congestion:31;Redness in ear:30;Pulling at ears:39;Fluid "
            "in ear:31;Coryza:42;"
            "Ear pain:38;Fever:31;Cough:45\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3,"
            "Abscess of nose,7,Nasal"
            " congestion:38;Irritable infant:49;Decreased appetite:37;Coryza"
            ":44;Ear pain:40;Fever"
            ":44;Cough:49\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3,"
            "Acute bronchospasm,8,"
            "Difficulty breathing:45;Nasal congestion:30;Vomiting:47;"
            "Shortness of breath:27;"
            "Wheezing:39;Coryza:44;Fever:31;Cough:28\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3,"
            "Abscess of the pharynx"
            ",5,Nasal congestion:32;Difficulty in swallowing:44;Ear pain:43;"
            "Fever:33;Cough:31\n"
            "cfd10d07-a121-4072-9fbb-8e080d794667,F,asian,nonhispanic,3,3,"
            "Acute sinusitis,8,Nasal"
            " congestion:25;Sore throat:44;Coughing up sputum:26;Painful "
            "sinuses:37;Coryza:42;Ear"
            " pain:35;Fever:26;Cough:45\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,"
            "Acute respiratory "
            "distress syndrome (ARDS),4,Difficulty breathing:46;Wheezing:47;"
            "Fever:30;Cough:45\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,Acute"
            " bronchiolitis,6,"
            "Nasal congestion:40;Vomiting:45;Pulling at ears:37;Wheezing:30;Fever"
            ":47;Cough:33\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,Acute"
            " otitis media,10,"
            "Nasal congestion:48;Sore throat:27;Redness in ear:45;Pulling at ears"
            ":47;Diminished "
            "hearing:39;Coryza:46;Ear pain:39;Fever:38;Cough:49;Plugged feeling"
            " in ear:32\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,"
            "Acariasis,3,Itching of "
            "skin:49;Vomiting:35;Skin rash:38\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,"
            "Abscess of the lung,4,"
            "Vomiting:28;Drainage in throat:35;Cough:32;Depressive or psychotic"
            " symptoms:32\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,Abscess"
            " of nose,5,Nasal "
            "congestion:39;Irritable infant:29;Coryza:32;Fever:34;Cough:48\n"
            "851fb57f-c2cf-49f9-b9c0-c3f55688b017,F,white,nonhispanic,1,1,Acute"
            " bronchospasm,6,"
            "Difficulty breathing:49;Nasal congestion:29;Shortness of breath:39;"
            "Wheezing:41;"
            "Fever:48;Cough:32\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acute "
            "bronchitis"
            ",7,Nasal congestion:27;Sore throat:45;Shortness of breath:32;Coughing"
            " up sputum"
            ":48;Wheezing:36;Fever:37;Cough:35\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acute "
            "bronchiolitis"
            ",11,Difficulty breathing:26;Nasal congestion:40;Irritable infant:"
            "30;Vomiting:49;"
            "Pulling at ears:37;Shortness of breath:35;Wheezing:33;Decreased "
            "appetite:39;Hurts"
            " to breath:25;Fever:42;Cough:43\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acute"
            " otitis media,8"
            ",Nasal congestion:32;Redness in ear:49;Pulling at ears:27;Fluid in"
            " ear:47;Diminished"
            " hearing:34;Ear pain:45;Fever:37;Cough:37\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acariasis"
            ",2,Vomiting:40;"
            "Skin rash:46\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acanthosis "
            "nigricans,2,Skin growth:40;Weight gain:25\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Abscess "
            "of nose,9,Nasal congestion:37;Irritable infant:33;Vomiting:37;Sinus "
            "congestion:37;Coryza:34;Ear pain:42;Fever:25;Cough:42;Abnormal breathing"
            " sounds:41\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acne,2,Skin"
            " rash:46;"
            "Acne or pimples:46\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Acute "
            "sinusitis,4,"
            "Nasal congestion:30;Ear pain:32;Fever:30;Cough:37\n"
            "ed9669df-d95a-42a8-b485-285dde7b998a,F,white,nonhispanic,0,0,Abdominal "
            "hernia,1,"
            "Infant spitting up:31\n"
            "109ea95d-1a96-4bc9-b8c9-8a56f5eb0a11,M,white,nonhispanic,34,34,Acute "
            "respiratory "
            "distress syndrome (ARDS),3,Difficulty breathing:35;Sharp chest pain:42"
            ";Shortness "
            "of breath:26\n"
            "109ea95d-1a96-4bc9-b8c9-8a56f5eb0a11,M,white,nonhispanic,34,34,"
            "Acute pancreatitis"
            ",3,Sharp chest pain:44;Vomiting:43;Sharp abdominal pain:35\n"
        )
        patients = tmpdir.join("patients.csv")
        patients.write(sample_data_patient)
        filename_patients = os.path.join(tmpdir, "patients.csv")

        sample_conditions = (
            "{\n"
            '    "acute-bronchitis": {\n'
            '        "condition_name": "Acute bronchitis",\n'
            '        "symptoms": {\n'
            '            "cough": {},\n'
            '            "nasal-congestion": {},\n'
            '            "fever": {},\n'
            '            "sore-throat": {},\n'
            '            "shortness-of-breath": {},\n'
            '            "coryza": {},\n'
            '            "sharp-chest-pain": {},\n'
            '            "coughing-up-sputum": {},\n'
            '            "difficulty-breathing": {},\n'
            '            "wheezing": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-fatty-liver-of-pregnancy-aflp": {\n'
            '        "condition_name": "Acute fatty liver of pregnancy (AFLP)",\n'
            '        "symptoms": {\n'
            '            "itching-of-skin": {},\n'
            '            "emotional-symptoms": {},\n'
            '            "cross-eyed": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-bronchiolitis": {\n'
            '        "condition_name": "Acute bronchiolitis",\n'
            '        "symptoms": {\n'
            '            "cough": {},\n'
            '            "fever": {},\n'
            '            "nasal-congestion": {},\n'
            '            "wheezing": {},\n'
            '            "difficulty-breathing": {},\n'
            '            "vomiting": {},\n'
            '            "coryza": {},\n'
            '            "shortness-of-breath": {},\n'
            '            "decreased-appetite": {},\n'
            '            "irritable-infant": {},\n'
            '            "pulling-at-ears": {},\n'
            '            "hurts-to-breath": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-otitis-media": {\n'
            '        "condition_name": "Acute otitis media",\n'
            '        "symptoms": {\n'
            '            "ear-pain": {},\n'
            '            "fever": {},\n'
            '            "cough": {},\n'
            '            "nasal-congestion": {},\n'
            '            "redness-in-ear": {},\n'
            '            "fluid-in-ear": {},\n'
            '            "sore-throat": {},\n'
            '            "coryza": {},\n'
            '            "pulling-at-ears": {},\n'
            '            "vomiting": {},\n'
            '            "diminished-hearing": {},\n'
            '            "plugged-feeling-in-ear": {}\n'
            "        }\n"
            "    },\n"
            '    "abscess-of-nose": {\n'
            '        "condition_name": "Abscess of nose",\n'
            '        "symptoms": {\n'
            '            "nasal-congestion": {},\n'
            '            "cough": {},\n'
            '            "fever": {},\n'
            '            "coryza": {},\n'
            '            "sore-throat": {},\n'
            '            "sinus-congestion": {},\n'
            '            "ear-pain": {},\n'
            '            "vomiting": {},\n'
            '            "abnormal-breathing-sounds": {},\n'
            '            "irritable-infant": {},\n'
            '            "decreased-appetite": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-bronchospasm": {\n'
            '        "condition_name": "Acute bronchospasm",\n'
            '        "symptoms": {\n'
            '            "cough": {},\n'
            '            "shortness-of-breath": {},\n'
            '            "wheezing": {},\n'
            '            "fever": {},\n'
            '            "difficulty-breathing": {},\n'
            '            "nasal-congestion": {},\n'
            '            "sharp-chest-pain": {},\n'
            '            "sore-throat": {},\n'
            '            "coryza": {},\n'
            '            "vomiting": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-sinusitis": {\n'
            '        "condition_name": "Acute sinusitis",\n'
            '        "symptoms": {\n'
            '            "cough": {},\n'
            '            "nasal-congestion": {},\n'
            '            "sore-throat": {},\n'
            '            "coryza": {},\n'
            '            "fever": {},\n'
            '            "ear-pain": {},\n'
            '            "sinus-congestion": {},\n'
            '            "painful-sinuses": {},\n'
            '            "coughing-up-sputum": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-respiratory-distress-syndrome-ards": {\n'
            '        "condition_name": "Acute respiratory distress syndrome (ARDS)",\n'
            '        "symptoms": {\n'
            '            "shortness-of-breath": {},\n'
            '            "difficulty-breathing": {},\n'
            '            "cough": {},\n'
            '            "sharp-chest-pain": {},\n'
            '            "depressive-or-psychotic-symptoms": {},\n'
            '            "fever": {},\n'
            '            "wheezing": {},\n'
            '            "hurts-to-breath": {},\n'
            '            "coughing-up-sputum": {}\n'
            "        }\n"
            "    },\n"
            '    "acariasis": {\n'
            '        "condition_name": "Acariasis",\n'
            '        "symptoms": {\n'
            '            "skin-rash": {},\n'
            '            "itching-of-skin": {},\n'
            '            "vomiting": {},\n'
            '            "cross-eyed": {},\n'
            '            "emotional-symptoms": {}\n'
            "        }\n"
            "    },\n"
            '    "abscess-of-the-lung": {\n'
            '        "condition_name": "Abscess of the lung",\n'
            '        "symptoms": {\n'
            '            "cough": {},\n'
            '            "sharp-chest-pain": {},\n'
            '            "shortness-of-breath": {},\n'
            '            "depressive-or-psychotic-symptoms": {},\n'
            '            "drainage-in-throat": {},\n'
            '            "vomiting": {}\n'
            "        }\n"
            "    },\n"
            '    "abscess-of-the-pharynx": {\n'
            '        "condition_name": "Abscess of the pharynx",\n'
            '        "symptoms": {\n'
            '            "sore-throat": {},\n'
            '            "fever": {},\n'
            '            "nasal-congestion": {},\n'
            '            "cough": {},\n'
            '            "difficulty-in-swallowing": {},\n'
            '            "ear-pain": {},\n'
            '            "sharp-chest-pain": {},\n'
            '            "coughing-up-sputum": {}\n'
            "        }\n"
            "    },\n"
            '    "acanthosis-nigricans": {\n'
            '        "condition_name": "Acanthosis nigricans",\n'
            '        "symptoms": {\n'
            '            "weight-gain": {},\n'
            '            "acne-or-pimples": {},\n'
            '            "skin-growth": {}\n'
            "        }\n"
            "    },\n"
            '    "acne": {\n'
            '        "condition_name": "Acne",\n'
            '        "symptoms": {\n'
            '            "acne-or-pimples": {},\n'
            '            "skin-rash": {},\n'
            '            "skin-growth": {}\n'
            "        }\n"
            "    },\n"
            '    "abdominal-hernia": {\n'
            '        "condition_name": "Abdominal hernia",\n'
            '        "symptoms": {\n'
            '            "sharp-abdominal-pain": {},\n'
            '            "infant-spitting-up": {}\n'
            "        }\n"
            "    },\n"
            '    "acute-pancreatitis": {\n'
            '        "condition_name": "Acute pancreatitis",\n'
            '        "symptoms": {\n'
            '            "sharp-abdominal-pain": {},\n'
            '            "vomiting": {},\n'
            '            "sharp-chest-pain": {}\n'
            "        }\n"
            "    }\n"
            "}\n"
        )
        conditions = tmpdir.join("conditions.json")
        conditions.write(sample_conditions)
        filename_conditions = os.path.join(tmpdir, "conditions.json")

        sample_symptoms = (
            "{\n"
            '    "sharp-chest-pain": {\n'
            '        "name": "Sharp chest pain"\n'
            "    },\n"
            '    "plugged-feeling-in-ear": {\n'
            '        "name": "Plugged feeling in ear"\n'
            "    },\n"
            '    "emotional-symptoms": {\n'
            '        "name": "Emotional symptoms"\n'
            "    },\n"
            '    "cross-eyed": {\n'
            '        "name": "Cross-eyed"\n'
            "    },\n"
            '    "itching-of-skin": {\n'
            '        "name": "Itching of skin"\n'
            "    },\n"
            '    "sore-throat": {\n'
            '        "name": "Sore throat"\n'
            "    },\n"
            '    "painful-sinuses": {\n'
            '        "name": "Painful sinuses"\n'
            "    },\n"
            '    "coughing-up-sputum": {\n'
            '        "name": "Coughing up sputum"\n'
            "    },\n"
            '    "redness-in-ear": {\n'
            '        "name": "Redness in ear"\n'
            "    },\n"
            '    "difficulty-in-swallowing": {\n'
            '        "name": "Difficulty in swallowing"\n'
            "    },\n"
            '    "fluid-in-ear": {\n'
            '        "name": "Fluid in ear"\n'
            "    },\n"
            '    "pulling-at-ears": {\n'
            '        "name": "Pulling at ears"\n'
            "    },\n"
            '    "vomiting": {\n'
            '        "name": "Vomiting"\n'
            "    },\n"
            '    "hurts-to-breath": {\n'
            '        "name": "Hurts to breath"\n'
            "    },\n"
            '    "depressive-or-psychotic-symptoms": {\n'
            '        "name": "Depressive or psychotic symptoms"\n'
            "    },\n"
            '    "sharp-abdominal-pain": {\n'
            '        "name": "Sharp abdominal pain"\n'
            "    },\n"
            '    "difficulty-breathing": {\n'
            '        "name": "Difficulty breathing"\n'
            "    },\n"
            '    "shortness-of-breath": {\n'
            '        "name": "Shortness of breath"\n'
            "    },\n"
            '    "infant-spitting-up": {\n'
            '        "name": "Infant spitting up"\n'
            "    },\n"
            '    "wheezing": {\n'
            '        "name": "Wheezing"\n'
            "    },\n"
            '    "nasal-congestion": {\n'
            '        "name": "Nasal congestion"\n'
            "    },\n"
            '    "decreased-appetite": {\n'
            '        "name": "Decreased appetite"\n'
            "    },\n"
            '    "ear-pain": {\n'
            '        "name": "Ear pain"\n'
            "    },\n"
            '    "diminished-hearing": {\n'
            '        "name": "Diminished hearing"\n'
            "    },\n"
            '    "cough": {\n'
            '        "name": "Cough"\n'
            "    },\n"
            '    "fever": {\n'
            '        "name": "Fever"\n'
            "    },\n"
            '    "skin-rash": {\n'
            '        "name": "Skin rash"\n'
            "    },\n"
            '    "acne-or-pimples": {\n'
            '        "name": "Acne or pimples"\n'
            "    },\n"
            '    "irritable-infant": {\n'
            '        "name": "Irritable infant"\n'
            "    },\n"
            '    "sinus-congestion": {\n'
            '        "name": "Sinus congestion"\n'
            "    },\n"
            '    "coryza": {\n'
            '        "name": "Coryza"\n'
            "    },\n"
            '    "abnormal-breathing-sounds": {\n'
            '        "name": "Abnormal breathing sounds"\n'
            "    },\n"
            '    "skin-growth": {\n'
            '        "name": "Skin growth"\n'
            "    },\n"
            '    "weight-gain": {\n'
            '        "name": "Weight gain"\n'
            "    },\n"
            '    "drainage-in-throat": {\n'
            '        "name": "Drainage in throat"\n'
            "    }\n"
            "}\n"
        )
        symptoms = tmpdir.join("symptoms.json")
        symptoms.write(sample_symptoms)
        filename_symptoms = os.path.join(tmpdir, "symptoms.json")

        env_kwargs = dict(
            id=GYM_ENV_ID,
            symptom_filepath=filename_symptoms,
            condition_filepath=filename_conditions,
            patient_filepath=filename_patients,
        )

        env = gym_make(**env_kwargs)

        env.reset()
        examples = self.get_random_example_from_env(env)

        factory = ReplayBufferFactory()
        replay_buf = factory.create(
            "UniformReplayBuffer",
            size=1000,
            B=1,
            example=self.examples_to_buffer(examples),
        )

        data = self.samples_to_buffer(self.get_random_example_from_env(env))
        data2 = self.samples_to_buffer(self.get_random_example_from_env(env))
        data3 = self.samples_to_buffer(self.get_random_example_from_env(env))

        replay_buf.append_samples(data)
        replay_buf.append_samples(data2)
        replay_buf.append_samples(data3)

        samples = replay_buf.sample_batch(1)

        assert isinstance(samples, AugSamplesFromReplay)
        assert torch.equal(samples.sim_patho, torch.from_numpy(data2.sim_patho[0, 0]))
        assert torch.equal(
            samples.sim_severity, torch.from_numpy(data2.sim_severity[0, 0])
        )
        assert torch.equal(
            samples.sim_evidence, torch.from_numpy(data2.sim_evidence[0, 0])
        )
        assert torch.equal(
            samples.sim_timestep, torch.from_numpy(data2.sim_timestep[0, 0])
        )
        assert torch.equal(
            samples[0].sim_patient, torch.from_numpy(data2.sim_patient[0, 0])
        )
        if data2.sim_differential_indices is None:
            assert samples.sim_differential_indices == data2.sim_differential_indices
        else:
            assert torch.equal(
                samples[0].sim_differential_indices,
                torch.from_numpy(data2.sim_differential_indices[0, 0]),
            )
        if data2.sim_differential_probas is None:
            assert samples.sim_differential_probas == data2.sim_differential_probas
        else:
            assert torch.equal(
                samples[0].sim_differential_probas,
                torch.from_numpy(data2.sim_differential_probas[0, 0]),
            )

        assert torch.equal(samples.action, torch.from_numpy(data2.action[0, 0]))
        assert torch.equal(samples.done, torch.from_numpy(data2.done[0, 0]))
        assert torch.equal(
            samples[0].agent_inputs.observation,
            torch.from_numpy(data2.observation[0, 0]),
        )
        assert torch.equal(
            samples[0].target_inputs.observation,
            torch.from_numpy(data3.observation[0, 0]),
        )

        # test intermediate data
        n_step_return = 3
        replay_buf2 = factory.create(
            "UniformReplayBuffer",
            size=1000,
            B=1,
            example=self.examples_to_buffer(examples),
            n_step_return=n_step_return,
            intermediate_data_flag=True,
        )
        env.reset()
        for _ in range(10 + n_step_return):
            data = self.samples_to_buffer(self.get_random_example_from_env(env))
            replay_buf2.append_samples(data)

        num_data = 4
        samples = replay_buf2.sample_batch(num_data)
        assert samples.intermediate_data is not None

        intermediate_data = samples.intermediate_data
        num_elts = (n_step_return - 1) * num_data
        assert intermediate_data.inputs.observation.size(0) == num_elts
        assert intermediate_data.inputs.prev_action.size(0) == num_elts
        assert intermediate_data.inputs.prev_reward.size(0) == num_elts
        assert intermediate_data.reward.size(0) == num_elts
        assert intermediate_data.done.size(0) == num_elts
        assert intermediate_data.done.ndim == 1
