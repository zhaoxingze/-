from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_configs_cover_required_experiments_and_extensions(self):
        from utils import load_config

        expected = {
            "linear_probe.yaml": ("E1", "linear_probe"),
            "partial_ft.yaml": ("E2", "partial_ft"),
            "full_ft.yaml": ("E3", "full_ft"),
            "ssf.yaml": ("E4", "ssf_all"),
            "ssf_l4.yaml": ("E5", "ssf_l4"),
            "ssf_l34.yaml": ("E6", "ssf_l34"),
            "fewshot_25.yaml": ("E7", "fewshot_25"),
            "fewshot_50.yaml": ("E8", "fewshot_50"),
            "lora.yaml": ("X1", "lora"),
            "adapter.yaml": ("X2", "adapter"),
        }

        for file_name, (experiment_id, method) in expected.items():
            with self.subTest(file_name=file_name):
                config = load_config(ROOT / "configs" / file_name)
                self.assertEqual(config["experiment_id"], experiment_id)
                self.assertEqual(config["method"], method)
                self.assertIn("purpose", config)
                self.assertIn("data", config)
                self.assertIn("training", config)

    def test_readme_matches_course_delivery_requirements(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        required_phrases = [
            "实验环境",
            "数据集下载",
            "运行方式",
            "实验结果",
            "E1",
            "E8",
            "LoRA",
            "Adapter",
        ]
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, readme)

    def test_expected_entry_points_exist(self):
        for relative_path in [
            "train.py",
            "evaluate.py",
            "dataset.py",
            "utils.py",
            "models/resnet_finetune.py",
            "models/ssf_resnet.py",
        ]:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((ROOT / relative_path).exists())

    def test_evaluate_formats_legacy_metrics_without_experiment_id(self):
        from evaluate import format_metric_row

        line = format_metric_row(
            {
                "method": "LINEAR",
                "test_acc_mean": "0.713",
                "trainable_percent": "0.0459",
            }
        )

        self.assertIn("-- LINEAR", line)
        self.assertIn("acc= 71.30%", line)


if __name__ == "__main__":
    unittest.main()
