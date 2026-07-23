from PIL import Image

from prooftag_qr.blueprints import canonical_url_match
from prooftag_qr.validation import Decoder, QRValidator


class FragmentDecoder(Decoder):
    name = "fragment"

    def decode(self, image: Image.Image) -> str:
        return "https://ptag.io/t/test#123456"


def test_validator_can_record_a_canonical_qart_contract_without_calling_it_exact():
    validator = QRValidator(decoders=[FragmentDecoder()])
    records = validator.validate(
        Image.new("RGB", (128, 128), "white"),
        "https://ptag.io/t/test",
        matcher=canonical_url_match,
        match_mode="canonical_url_without_fragment",
    )

    assert records
    assert all(record.exact_payload_match for record in records)
    assert all(
        record.parameters["match_mode"] == "canonical_url_without_fragment"
        for record in records
    )
