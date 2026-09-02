from unittest import TestCase
from unittest.mock import patch

from scx_router import SCXRouter


class SCXRouterTest(TestCase):
    @patch("scx_router.router.ZeroShotClassificationPipeline")
    @patch("scx_router.router.AutoTokenizer.from_pretrained")
    @patch("scx_router.router.GLiClassModel.from_pretrained")
    def test_rank_returns_descending_scores(self, load_model, load_tokenizer, pipeline_type):
        pipeline_type.return_value.return_value = [
            [
                {"label": "small", "score": 0.2},
                {"label": "large", "score": 0.9},
            ]
        ]

        router = SCXRouter(device="cpu")
        result = router.rank("Solve this proof.", ["small", "large"])

        self.assertEqual([item["label"] for item in result], ["large", "small"])
        pipeline_type.return_value.assert_called_once_with(
            "Solve this proof.", ["small", "large"], threshold=0.0
        )
        load_model.assert_called_once()
        load_tokenizer.assert_called_once()

    @patch("scx_router.router.ZeroShotClassificationPipeline")
    @patch("scx_router.router.AutoTokenizer.from_pretrained")
    @patch("scx_router.router.GLiClassModel.from_pretrained")
    def test_rank_rejects_empty_input(self, _load_model, _load_tokenizer, _pipeline_type):
        router = SCXRouter(device="cpu")

        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            router.rank("  ")

        with self.assertRaisesRegex(ValueError, "labels must not be empty"):
            router.rank("hello", [])
