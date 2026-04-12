-- Enable pgvector
create extension if not exists vector;

-- Clinical guidelines table
create table clinical_guidelines (
    id bigserial primary key,
    content_hash text unique,
    source_org text,
    disease_topic text,
    url_reference text,
    chunk_content text,
    embedding vector(768)
);

-- Drug labels table
create table drug_labels (
    id bigserial primary key,
    content_hash text unique,
    drug_name text,
    indication text,
    dosage_and_administration text,
    warnings_and_precautions text,
    renal_adjustment text,
    chunk_content text,
    embedding vector(768)
);

-- Vector search function for guidelines
create or replace function match_guidelines(
    query_embedding vector(768),
    match_threshold float,
    match_count int
)
returns table (
    id bigint, source_org text, disease_topic text,
    url_reference text, chunk_content text, similarity float
)
language sql stable as $$
    select id, source_org, disease_topic, url_reference, chunk_content,
           1 - (embedding <=> query_embedding) as similarity
    from clinical_guidelines
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by similarity desc
    limit match_count;
$$;

-- Vector search function for drug labels
create or replace function match_drug_labels(
    query_embedding vector(768),
    match_threshold float,
    match_count int
)
returns table (
    id bigint, drug_name text, chunk_content text, similarity float
)
language sql stable as $$
    select id, drug_name, chunk_content,
           1 - (embedding <=> query_embedding) as similarity
    from drug_labels
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by similarity desc
    limit match_count;
$$;