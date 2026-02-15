# src/utils/

**Purpose:** Shared utilities: YAML load, date helpers, data validation, logging. No business logic; used across pipeline, inference, and app_services.

**Fits into system:** Called by config loaders, pipeline scripts, and evaluation. Pure helpers.

**Data in/out:**
- **In:** File paths, config dicts, raw data for validation.
- **Out:** Parsed config, validated flags, normalized dates.
