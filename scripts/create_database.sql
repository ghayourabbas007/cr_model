-- One-time MySQL setup. Run as an admin user, e.g.:
--   mysql -u root -p < scripts/create_database.sql
-- Change the password, then put the same value in .env (DB_PASSWORD).

CREATE DATABASE IF NOT EXISTS crypto_forecast
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

CREATE USER IF NOT EXISTS 'forecaster'@'localhost' IDENTIFIED BY 'change-me';
GRANT ALL PRIVILEGES ON crypto_forecast.* TO 'forecaster'@'localhost';
FLUSH PRIVILEGES;
