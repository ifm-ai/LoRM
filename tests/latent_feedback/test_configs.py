from lorm.train.main import resolve_config
from works.latent_feedback.make_config import GPUS, make_config, runs


def test_every_run_resolves_and_branches_from_a_listed_trunk():
    table = runs()
    assert len(table) == 195 and sum(row["released"] == "1" for row in table.values()) == 174
    for name, row in table.items():
        config = resolve_config(make_config(row))
        assert config["train"]["batch_size"] * config["train"]["gradient_accumulation"] * GPUS[row["size"]] == int(row["sequences"])
        if row["trunk"]:
            trunk = table[row["trunk"]]
            assert int(row["branch_step"]) in [int(s) for s in trunk["branch_points"].split()], name
            assert (trunk["size"], trunk["arm"], trunk["upn"], trunk["seed"]) == (row["size"], row["arm"], row["upn"],
                                                                                  row["seed"]), name
