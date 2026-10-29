from app.core.config import Settings


def test_llm_keys_by_field_name_and_by_mycel_env_names(monkeypatch):
    assert Settings(_env_file=None, groq_api_keys="a, b", gemini_api_keys=["g"]).groq_api_keys == ["a", "b"]

    monkeypatch.setenv("GROQ_EXTREME_POOL", "k1,k2,k3")
    monkeypatch.setenv("GEMINI_API_KEY", "g1")
    loaded = Settings(_env_file=None)
    assert loaded.groq_api_keys == ["k1", "k2", "k3"] and loaded.gemini_api_keys == ["g1"]


def test_list_settings_are_comma_separated_in_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test,http://b.test")
    monkeypatch.setenv("AUTO_APPROVE_CATEGORIES", "install")
    monkeypatch.setenv("ALLOWED_REPO_ROOTS", str(tmp_path))
    loaded = Settings(_env_file=None)
    assert loaded.cors_origins == ["http://a.test", "http://b.test"]
    assert loaded.auto_approve_categories == ["install"] and loaded.allowed_repo_roots == [tmp_path.resolve()]
