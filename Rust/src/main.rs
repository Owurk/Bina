use std::sync::{Arc, Mutex};
use axum::{extract::State, routing::post, Json, Router};
use serde_json::{json, Value};

type Events = Arc<Mutex<Vec<Value>>>;

async fn ingest_event(
    State(events): State<Events>,
    Json(event): Json<Value>,
) -> Json<Value> {
    events.lock().unwrap().push(event);
    Json(json!({"status": "ok"}))
}

#[tokio::main]
async fn main() {
    let events: Events = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/events", post(ingest_event))
        .with_state(events);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind port 3000");
    
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.expect("Server error");
}

