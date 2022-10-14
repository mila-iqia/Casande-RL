from rlpyt.agents.base import AlternatingRecurrentAgentMixin
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.pg.categorical import (
    CategoricalPgAgent,
    RecurrentCategoricalPgAgentBase,
)
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.pg.base import PolicyGradientAlgo
from rlpyt.envs.gym import make as gym_make

from chloe.utils.agent_utils import AgentFactory
from chloe.utils.algo_utils import AlgoFactory
from chloe.utils.runner_utils import RunnerFactory
from chloe.utils.sampler_utils import (
    SamplerFactory,
    is_async_sampler,
    is_aternating_sampler,
)
from chloe.utils.train_utils import (
    check_run_id_file_existence,
    load_run_ids,
    save_run_ids,
)


class TestTrainUtils(object):
    def test_save_load_and_check_run_ids(self, tmpdir):

        mlflow_run_id = "abcd"
        logging_run_id = "xyz1"

        save_run_ids(tmpdir, mlflow_run_id, logging_run_id)

        assert check_run_id_file_existence(tmpdir)

        mlflow_run_id_1, logging_run_id_1 = load_run_ids(tmpdir)

        assert mlflow_run_id == mlflow_run_id_1
        assert logging_run_id == logging_run_id_1


class TestSamplerUtils(object):
    def test_sampler_factory(self):

        factory = SamplerFactory()

        list_samplers = [
            "serialsampler",
            "cpusampler",
            "gpusampler",
            "alternatingsampler",
            "nooverlapalternatingsampler",
            "asyncserialsampler",
            "asynccpusampler",
            "asyncgpusampler",
            "asyncalternatingsampler",
            "asyncnooverlapalternatingsampler",
        ]
        for sampler_type in list_samplers:
            sampler = factory.create(
                sampler_type,
                EnvCls=gym_make,
                env_kwargs=dict(id="MountainCar-v0"),
                eval_env_kwargs=dict(id="MountainCar-v0"),
                batch_T=1,
                batch_B=2,
                max_decorrelation_steps=0,
                eval_n_envs=10,
                eval_max_steps=10,
                eval_max_trajectories=50,
            )
            if "alternating" in sampler_type.lower():
                assert is_aternating_sampler(sampler)
            else:
                assert not is_aternating_sampler(sampler)

            if "async" in sampler_type.lower():
                assert is_async_sampler(sampler)
            else:
                assert not is_async_sampler(sampler)

        try:
            factory.create(
                "randomkeysampler",
                EnvCls=gym_make,
                env_kwargs=dict(id="MountainCar-v0"),
                eval_env_kwargs=dict(id="MountainCar-v0"),
                batch_T=1,
                batch_B=2,
                max_decorrelation_steps=0,
                eval_n_envs=10,
                eval_max_steps=10,
                eval_max_trajectories=50,
            )
            assert False
        except Exception:
            assert True


class TestAlgoUtils(object):
    def test_algo_factory(self):

        factory = AlgoFactory()

        list_algos = [
            "a2c",
            "ppo",
            "dqn",
            "categoricaldqn",
            "r2d1",
        ]
        for algo_type in list_algos:
            _ = factory.create(algo_type)
        assert True

        try:
            factory.create("randomkeyalgo",)
            assert False
        except Exception:
            assert True


class TestAgentUtils(object):
    def test_agent_factory(self):

        factory = AgentFactory()
        sampler_factory = SamplerFactory()
        algo_factory = AlgoFactory()

        list_agents = [
            "alternatingrecurrentcategoricalpgagent",
            "r2d1alternatingagent",
            "categoricalpgagent",
            "recurrentcategoricalpgagent",
            "dqnagent",
            "catdqnagent",
            "r2d1agent",
        ]
        list_algos = [
            "a2c",
            "ppo",
            "dqn",
            "categoricaldqn",
            "r2d1",
        ]
        list_samplers = [
            "serialsampler",
            "cpusampler",
            "gpusampler",
            "alternatingsampler",
            "nooverlapalternatingsampler",
            "asyncserialsampler",
            "asynccpusampler",
            "asyncgpusampler",
            "asyncalternatingsampler",
            "asyncnooverlapalternatingsampler",
        ]
        for sampler_type in list_samplers:
            sampler = sampler_factory.create(
                sampler_type,
                EnvCls=gym_make,
                env_kwargs=dict(id="MountainCar-v0"),
                eval_env_kwargs=dict(id="MountainCar-v0"),
                batch_T=1,
                batch_B=2,
                max_decorrelation_steps=0,
                eval_n_envs=10,
                eval_max_steps=10,
                eval_max_trajectories=50,
            )
            sampler_cls = sampler.__class__
            is_alternating_sampler = is_aternating_sampler(sampler)
            for algo_type in list_algos:
                algo = algo_factory.create(algo_type)
                algo_cls = algo.__class__
                is_pg_algo = isinstance(algo, PolicyGradientAlgo)
                is_dqn_algo = isinstance(algo, DQN)
                for agent_type in list_agents:
                    is_dqn_agent = issubclass(
                        factory.get_agent_class(agent_type), DqnAgent
                    )
                    is_pg_agent = issubclass(
                        factory.get_agent_class(agent_type),
                        (CategoricalPgAgent, RecurrentCategoricalPgAgentBase),
                    )
                    is_alternating_agent = issubclass(
                        factory.get_agent_class(agent_type),
                        AlternatingRecurrentAgentMixin,
                    )
                    pg_match = (is_pg_agent and is_pg_algo) or (
                        not is_pg_agent and not is_pg_algo
                    )
                    dqn_match = (is_dqn_agent and is_dqn_algo) or (
                        not is_dqn_agent and not is_dqn_algo
                    )
                    alternating_match = (
                        is_alternating_agent and is_alternating_sampler
                    ) or (not is_alternating_agent and not is_alternating_sampler)

                    type_match = pg_match and dqn_match and alternating_match

                    try:
                        factory.create(
                            agent_type, algo_cls, sampler_cls,
                        )
                        assert type_match
                    except Exception:
                        assert not type_match


class TestRunnerUtils(object):
    def test_runner_factory(self):

        factory = RunnerFactory()
        sampler_factory = SamplerFactory()

        kwargs = dict(
            EnvCls=gym_make,
            env_kwargs=dict(id="MountainCar-v0"),
            eval_env_kwargs=dict(id="MountainCar-v0"),
            batch_T=1,
            batch_B=2,
            max_decorrelation_steps=0,
            eval_n_envs=10,
            eval_max_steps=10,
            eval_max_trajectories=50,
        )

        list_runners = [
            "minibatchrl",
            "minibatchrleval",
            "syncrl",
            "syncrleval",
            "asyncrl",
            "asyncrleval",
        ]
        list_samplers = [
            "serialsampler",
            "cpusampler",
            "gpusampler",
            "alternatingsampler",
            "nooverlapalternatingsampler",
            "asyncserialsampler",
            "asynccpusampler",
            "asyncgpusampler",
            "asyncalternatingsampler",
            "asyncnooverlapalternatingsampler",
        ]

        for sampler_type in list_samplers:
            sampler = sampler_factory.create(sampler_type, **kwargs)
            for runner_type in list_runners:
                if "minibatch" in runner_type.lower():
                    if "async" in sampler_type.lower():
                        try:
                            factory.get_affinity(
                                runner_type, sampler, cuda_idx=0, n_workers=2, n_gpus=1
                            )
                            assert False
                        except Exception:
                            assert True
                else:
                    is_runner_async = "async" in runner_type.lower()
                    is_sampler_async = "async" in sampler_type.lower()

                    async_combination = is_runner_async and is_sampler_async
                    sync_combination = not is_runner_async and not is_sampler_async
                    is_valid_combination = async_combination or sync_combination

                    if not is_valid_combination:
                        try:
                            factory.get_affinity(
                                runner_type, sampler, cuda_idx=0, n_workers=2, n_gpus=1
                            )
                            assert False
                        except Exception:
                            assert True
