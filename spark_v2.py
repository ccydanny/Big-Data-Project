# spark_top_words_final.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, split, lower, regexp_replace, trim, when, length,
    desc, lit
)

# ---- CONFIG ----
INPUT  = "s3://amazonreviews3bucket/input/train.csv"
OUT_POS = "s3://amazonreviews3bucket/output/top30_positive"
OUT_NEG = "s3://amazonreviews3bucket/output/top30_negative"
OUT_ALL = "s3://amazonreviews3bucket/output/top30_words_both"

spark = (SparkSession.builder
         .appName("Top30Words_PositiveNegative")
         .getOrCreate())

# ---- 1) Load exactly 3 columns: polarity, title, text ----
df_raw = spark.read.csv(INPUT, header=False, inferSchema=False)
# Expecting _c0=polarity, _c1=title, _c2=text
df = (df_raw
      .select(
          col("_c0").alias("polarity"),
          col("_c1").alias("title"),
          col("_c2").alias("text")
      ))

# ---- 2) Map polarity: 1 = negative, 2 = positive ----
p = trim(lower(col("polarity")))
df = df.withColumn(
    "label",
    when(p == "2", 1)  # positive
    .when(p == "1", 0) # negative
    .otherwise(None)
)

# Keep only rows with a valid label and non-empty text
df = df.filter(df.label.isNotNull() & df.text.isNotNull() & (trim(col("text")) != ""))

# ---- 3) Clean + tokenize text ----
# Keep letters/spaces; lowercase; split; drop empties and 1-char tokens
df = df.withColumn("text", regexp_replace(col("text"), r"[^A-Za-z\s]", " "))
df = df.withColumn("text", lower(trim(col("text"))))
words = (df.withColumn("word", explode(split(col("text"), r"\s+")))
           .filter((col("word") != "") & col("word").isNotNull() & (length(col("word")) >= 2)))

# ---- 4) Aggregate top 30 by sentiment ----
topN = 30
pos = (words.filter(col("label") == 1)
            .groupBy("word").count()
            .orderBy(desc("count")).limit(topN).coalesce(1))
neg = (words.filter(col("label") == 0)
            .groupBy("word").count()
            .orderBy(desc("count")).limit(topN).coalesce(1))

# ---- 5) Quick sanity prints to step logs ----
print("Counts by normalized label (0=neg, 1=pos):")
df.groupBy("label").count().show()

print("Preview POS:"); pos.show(10, truncate=False)
print("Preview NEG:"); neg.show(10, truncate=False)

# ---- 6) Write outputs (CSV with header) ----
if pos.count() > 0:
    pos.write.mode("overwrite").option("header", "true").csv(OUT_POS)
else:
    print("WARNING: No positive rows; nothing written to", OUT_POS)

if neg.count() > 0:
    neg.write.mode("overwrite").option("header", "true").csv(OUT_NEG)
else:
    print("WARNING: No negative rows; nothing written to", OUT_NEG)

# combined (easier to view)
pos_tag = pos.withColumn("sentiment", lit("positive"))
neg_tag = neg.withColumn("sentiment", lit("negative"))
combined = pos_tag.unionByName(neg_tag, allowMissingColumns=True) \
                  .select("sentiment", "word", "count").coalesce(1)
if combined.count() > 0:
    combined.write.mode("overwrite").option("header", "true").csv(OUT_ALL)
else:
    print("WARNING: Combined output empty; nothing written to", OUT_ALL)

spark.stop()
