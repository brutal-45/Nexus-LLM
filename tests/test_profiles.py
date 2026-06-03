"""Test configuration profiles for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from copy import deepcopy


@dataclass
class Profile:
    name: str
    description: str
    config: Dict[str, Any]


# Built-in profiles
PROFILES: Dict[str, Profile] = {
    "default": Profile(
        name="default",
        description="Default configuration for general use",
        config={
            "model": {"name": "nexus-llm-base", "temperature": 0.7, "top_p": 0.9, "max_length": 2048},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "safety": {"enabled": True, "max_toxicity": 0.5},
        },
    ),
    "creative": Profile(
        name="creative",
        description="Creative writing with higher temperature",
        config={
            "model": {"name": "nexus-llm-base", "temperature": 1.2, "top_p": 0.95, "max_length": 4096},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "safety": {"enabled": True, "max_toxicity": 0.5},
        },
    ),
    "precise": Profile(
        name="precise",
        description="Precise responses with low temperature",
        config={
            "model": {"name": "nexus-llm-base", "temperature": 0.1, "top_p": 0.5, "max_length": 1024},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "safety": {"enabled": True, "max_toxicity": 0.3},
        },
    ),
    "code": Profile(
        name="code",
        description="Code generation profile",
        config={
            "model": {"name": "nexus-llm-code", "temperature": 0.2, "top_p": 0.8, "max_length": 4096},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "safety": {"enabled": True, "max_toxicity": 0.3},
        },
    ),
    "fast": Profile(
        name="fast",
        description="Fast inference with minimal safety",
        config={
            "model": {"name": "nexus-llm-small", "temperature": 0.7, "top_p": 0.9, "max_length": 512},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 8},
            "safety": {"enabled": False, "max_toxicity": 1.0},
        },
    ),
}


class ProfileManager:
    def __init__(self):
        self._profiles = deepcopy(PROFILES)
        self._active = "default"

    def get(self, name: str) -> Profile:
        if name not in self._profiles:
            raise KeyError(f"Profile '{name}' not found")
        return deepcopy(self._profiles[name])

    def list_profiles(self) -> list:
        return list(self._profiles.keys())

    def add_profile(self, profile: Profile) -> None:
        self._profiles[profile.name] = profile

    def remove_profile(self, name: str) -> None:
        if name not in self._profiles:
            raise KeyError(f"Profile '{name}' not found")
        if name == "default":
            raise ValueError("Cannot remove the default profile")
        del self._profiles[name]

    def set_active(self, name: str) -> None:
        if name not in self._profiles:
            raise KeyError(f"Profile '{name}' not found")
        self._active = name

    def get_active(self) -> Profile:
        return self.get(self._active)

    def merge(self, base_name: str, overrides: Dict[str, Any]) -> Profile:
        base = self.get(base_name)
        merged_config = deepcopy(base.config)
        for key, value in overrides.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        return Profile(name=base_name + "_merged", description=f"Merged from {base_name}", config=merged_config)


class TestProfileDefinitions:
    def test_default_profile_exists(self):
        assert "default" in PROFILES

    def test_creative_profile_exists(self):
        assert "creative" in PROFILES

    def test_precise_profile_exists(self):
        assert "precise" in PROFILES

    def test_code_profile_exists(self):
        assert "code" in PROFILES

    def test_fast_profile_exists(self):
        assert "fast" in PROFILES

    def test_all_profiles_have_name(self):
        for name, profile in PROFILES.items():
            assert profile.name == name

    def test_all_profiles_have_description(self):
        for profile in PROFILES.values():
            assert profile.description
            assert isinstance(profile.description, str)

    def test_all_profiles_have_config(self):
        for profile in PROFILES.values():
            assert isinstance(profile.config, dict)
            assert "model" in profile.config
            assert "api" in profile.config


class TestProfileManager:
    def test_get_default_profile(self):
        pm = ProfileManager()
        profile = pm.get("default")
        assert profile.name == "default"
        assert profile.config["model"]["temperature"] == 0.7

    def test_get_nonexistent_profile(self):
        pm = ProfileManager()
        with pytest.raises(KeyError, match="not found"):
            pm.get("nonexistent")

    def test_list_profiles(self):
        pm = ProfileManager()
        profiles = pm.list_profiles()
        assert "default" in profiles
        assert "creative" in profiles
        assert len(profiles) >= 5

    def test_add_profile(self):
        pm = ProfileManager()
        new_profile = Profile(
            name="custom",
            description="Custom profile",
            config={"model": {"temperature": 0.5}},
        )
        pm.add_profile(new_profile)
        assert "custom" in pm.list_profiles()
        assert pm.get("custom").config["model"]["temperature"] == 0.5

    def test_remove_profile(self):
        pm = ProfileManager()
        pm.remove_profile("fast")
        assert "fast" not in pm.list_profiles()

    def test_remove_default_fails(self):
        pm = ProfileManager()
        with pytest.raises(ValueError, match="Cannot remove"):
            pm.remove_profile("default")

    def test_remove_nonexistent_fails(self):
        pm = ProfileManager()
        with pytest.raises(KeyError, match="not found"):
            pm.remove_profile("nonexistent")

    def test_set_active(self):
        pm = ProfileManager()
        pm.set_active("creative")
        active = pm.get_active()
        assert active.name == "creative"

    def test_set_active_nonexistent(self):
        pm = ProfileManager()
        with pytest.raises(KeyError, match="not found"):
            pm.set_active("nonexistent")

    def test_get_returns_copy(self):
        pm = ProfileManager()
        p1 = pm.get("default")
        p1.config["model"]["temperature"] = 99.0
        p2 = pm.get("default")
        assert p2.config["model"]["temperature"] == 0.7

    def test_merge_overrides(self):
        pm = ProfileManager()
        merged = pm.merge("default", {"model": {"temperature": 1.5}})
        assert merged.config["model"]["temperature"] == 1.5
        assert merged.config["model"]["top_p"] == 0.9

    def test_merge_adds_new_key(self):
        pm = ProfileManager()
        merged = pm.merge("default", {"model": {"custom_flag": True}})
        assert merged.config["model"]["custom_flag"] is True


class TestProfileCharacteristics:
    def test_creative_has_higher_temperature(self):
        assert PROFILES["creative"].config["model"]["temperature"] > PROFILES["default"].config["model"]["temperature"]

    def test_precise_has_lower_temperature(self):
        assert PROFILES["precise"].config["model"]["temperature"] < PROFILES["default"].config["model"]["temperature"]

    def test_code_profile_type(self):
        assert "code" in PROFILES["code"].config["model"]["name"]

    def test_fast_profile_has_safety_disabled(self):
        assert PROFILES["fast"].config["safety"]["enabled"] is False

    def test_fast_profile_more_workers(self):
        assert PROFILES["fast"].config["api"]["workers"] > PROFILES["default"].config["api"]["workers"]
