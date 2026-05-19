"""Tests for nexus_llm.protocols.training_protocol module."""

import pytest
from nexus_llm.protocols.training_protocol import TrainingRequest, TrainingResponse, TrainingProtocol


class SampleTrainingProtocol(TrainingProtocol):
    def train(self, request):
        return TrainingResponse(job_id="job-123", status="started")


class TestTrainingRequest:
    def test_default(self):
        req = TrainingRequest()
        assert req.epochs == 3
        assert req.learning_rate == 5e-5

    def test_with_config(self):
        req = TrainingRequest(model="gpt-4", dataset="my_data", epochs=10, learning_rate=1e-4)
        assert req.epochs == 10


class TestTrainingResponse:
    def test_default(self):
        resp = TrainingResponse()
        assert resp.status == "pending"

    def test_with_job(self):
        resp = TrainingResponse(job_id="job-1", status="running")
        assert resp.job_id == "job-1"


class TestTrainingProtocol:
    def test_validate_valid(self):
        proto = SampleTrainingProtocol()
        req = TrainingRequest(model="gpt-4", dataset="data")
        errors = proto.validate_request(req)
        assert errors == []

    def test_validate_no_model(self):
        proto = SampleTrainingProtocol()
        req = TrainingRequest(dataset="data")
        errors = proto.validate_request(req)
        assert len(errors) > 0

    def test_train(self):
        proto = SampleTrainingProtocol()
        req = TrainingRequest(model="gpt-4", dataset="data")
        resp = proto.train(req)
        assert resp.status == "started"
