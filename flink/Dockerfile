FROM flink:1.20-java17

# Connectors are not part of Flink itself. They are added from Maven Central.
# The jars must be readable by the `flink` user, hence the explicit mode.
ARG MAVEN=https://repo1.maven.org/maven2
ADD --chmod=644 ${MAVEN}/org/apache/flink/flink-sql-connector-kafka/3.4.0-1.20/flink-sql-connector-kafka-3.4.0-1.20.jar /opt/flink/lib/
ADD --chmod=644 ${MAVEN}/org/apache/flink/flink-connector-jdbc/3.3.0-1.20/flink-connector-jdbc-3.3.0-1.20.jar /opt/flink/lib/
ADD --chmod=644 ${MAVEN}/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar /opt/flink/lib/
