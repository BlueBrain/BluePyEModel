"""Tests for EModelMetadata methods."""

import pytest

from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata


metadata_args = {
    "emodel": "L5_TPC:B_cAC",
    "etype": "cAC",
    "mtype": "L5_TPC:B",
    "ttype": "245_L5 PT CTX",
    "species": "mouse",
    "brain_region": "SSCX",
    "iteration_tag": "v0",
    "morph_class": "PYR",
    "synapse_class": "EXC",
    "layer": "L5",
}


@pytest.fixture
def metadata():
    return EModelMetadata(**metadata_args)


def test_init():
    """Test constructor."""
    with pytest.raises(ValueError, match="At least emodel or etype should be informed"):
        metadata = EModelMetadata()
    with pytest.raises(ValueError, match="morph_class should be 'PYR' or 'INT'"):
        metadata = EModelMetadata(emodel="L5_TPC:B_cAC", morph_class="bad morph class")
    with pytest.raises(ValueError, match="synapse_class should be 'EXC' or 'INH'"):
        metadata = EModelMetadata(emodel="L5_TPC:B_cAC", synapse_class="bad synapse class")

    metadata = EModelMetadata(**metadata_args)

    assert metadata.emodel == "L5_TPC:B_cAC"
    assert metadata.etype == "cAC"
    assert metadata.mtype == "L5_TPC:B"
    assert metadata.ttype == "245_L5 PT CTX"
    assert metadata.species == "mouse"
    assert metadata.brain_region == "SSCX"
    assert metadata.iteration == "v0"
    assert metadata.morph_class == "PYR"
    assert metadata.synapse_class == "EXC"
    assert metadata.layer == "L5"


def test_etype_annotation_dict(metadata):
    """Test etype_annotation_dict method."""
    assert metadata.etype_annotation_dict() == {
        "type": [
            "ETypeAnnotation",
            "Annotation",
        ],
        "hasBody": {
            "type": [
                "EType",
                "AnnotationBody",
            ],
            "label": "cAC",
        },
        "name": "E-type annotation",
    }


def test_mtype_annotation_dict(metadata):
    """Test mtype_annotation_dict method."""
    assert metadata.mtype_annotation_dict() == {
        "type": [
            "MTypeAnnotation",
            "Annotation",
        ],
        "hasBody": {
            "type": [
                "MType",
                "AnnotationBody",
            ],
            "label": "L5_TPC:B",
        },
        "name": "M-type annotation",
    }


def test_ttype_annotation_dict(metadata):
    """Test ttype_annotation_dict method."""
    assert metadata.ttype_annotation_dict() == {
        "type": [
            "TTypeAnnotation",
            "Annotation",
        ],
        "hasBody": {
            "type": [
                "TType",
                "AnnotationBody",
            ],
            "label": "245_L5 PT CTX",
        },
        "name": "T-type annotation",
    }


def test_annotation_list(metadata):
    """Test annotation_list method."""
    annotation_list = metadata.annotation_list()
    assert len(annotation_list) == 3
    assert annotation_list[0]["hasBody"]["label"] == "cAC"
    assert annotation_list[1]["hasBody"]["label"] == "L5_TPC:B"
    assert annotation_list[2]["hasBody"]["label"] == "245_L5 PT CTX"

    # test when some of etype, mtype or ttype are None
    metadata_args_2 = metadata_args.copy()
    metadata_args_2["etype"] = None
    metadata_args_2["mtype"] = None
    metadata_2 = EModelMetadata(**metadata_args_2)
    annotation_list_2 = metadata_2.annotation_list()
    assert len(annotation_list_2) == 1
    assert annotation_list_2[0]["hasBody"]["label"] == "245_L5 PT CTX"


def test_get_metadata_dict(metadata):
    """Test get_metadata_dict method."""
    metadata_dict = metadata_args.copy()
    metadata_dict["subject"] = metadata_dict.pop("species")
    metadata_dict["brainLocation"] = metadata_dict.pop("brain_region")
    metadata_dict["iteration"] = metadata_dict.pop("iteration_tag")

    assert metadata.get_metadata_dict() == metadata_dict

    # with None values
    metadata = EModelMetadata(emodel="test")
    assert metadata.get_metadata_dict() == {"emodel": "test"}


def test_filters_for_resource(metadata):
    """Test filters_for_resource method."""
    metadata_dict = metadata_args.copy()
    metadata_dict["subject"] = metadata_dict.pop("species")
    metadata_dict["brainLocation"] = metadata_dict.pop("brain_region")
    metadata_dict["iteration"] = metadata_dict.pop("iteration_tag")

    assert metadata.get_metadata_dict() == metadata_dict

    # with None values
    metadata = EModelMetadata(emodel="test")
    assert metadata.get_metadata_dict() == {"emodel": "test"}


def test_for_resource(metadata):
    """Test for_resource method."""
    metadata_dict = metadata_args.copy()
    metadata_dict["subject"] = metadata_dict.pop("species")
    metadata_dict["brainLocation"] = metadata_dict.pop("brain_region")
    metadata_dict["iteration"] = metadata_dict.pop("iteration_tag")
    metadata_dict["annotation"] = metadata.annotation_list()

    assert metadata.for_resource() == metadata_dict

    # with None values
    metadata = EModelMetadata(emodel="test", mtype="mtype_test")
    assert metadata.for_resource() == {
        "emodel": "test",
        "mtype": "mtype_test",
        "annotation": [
            {
                "type": [
                    "MTypeAnnotation",
                    "Annotation",
                ],
                "hasBody": {
                    "type": [
                        "MType",
                        "AnnotationBody",
                    ],
                    "label": "mtype_test",
                },
                "name": "M-type annotation",
            }
        ],
    }


def test_as_string(metadata):
    """Test as_string method."""
    assert metadata.as_string(seed=42) == (
        "emodel=L5_TPC:B_cAC__etype=cAC__ttype=245_L5 PT CTX__mtype=L5_TPC:B__"
        "species=mouse__brain_region=SSCX__iteration=v0__seed=42"
    )

    # with None values and slashes
    metadata = EModelMetadata(emodel="w/it/h_sla/she/s")
    assert metadata.as_string(seed="None") == "emodel=with_slashes"