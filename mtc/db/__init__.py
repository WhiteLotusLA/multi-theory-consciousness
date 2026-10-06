"""
Database Setup
==============

Store setup that a non-editable install can run from any directory:

- ``mtc.db.migrations`` renders ``migrations/postgres/*.sql`` from the schema
  registry (``mtc.core.schema_registry.SCHEMA_ARTIFACTS``);
- ``mtc.db.postgres_store``, ``mongo_store``, ``qdrant_store``,
  ``neo4j_store`` and ``redis_store`` probe, initialize and check one store
  each, returning ``mtc.core.checks.CheckResult`` values;
- ``mtc.db.bootstrap`` initializes every enabled store, records the install
  date and seeds the stores (``mtc db init``);
- ``mtc.db.verify`` checks every enabled store against the registry
  (``mtc db check``);
- ``mtc.db.reembed`` backfills Qdrant points for PostgreSQL episodic rows.

Importing this package or any of its modules loads no store client, web
framework or HTTP client; the clients come from ``mtc.memory.clients`` inside
the calls that need them. Nothing here drops, truncates, deletes or recreates
a database, table, collection, graph or key.
"""
