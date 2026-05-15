//! `lumen` — command-line entry point.
//!
//! Subcommands (planned):
//!   lumen run <model.gguf> "prompt"
//!   lumen compile <input.lum> -o out.o
//!   lumen bench <model.gguf>

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => {
            eprintln!("lumen run: not implemented yet (Phase 6)");
            ExitCode::from(2)
        }
        Some("compile") => {
            eprintln!("lumen compile: not implemented yet (Phase 2)");
            ExitCode::from(2)
        }
        Some("bench") => {
            eprintln!("lumen bench: not implemented yet (Phase 7)");
            ExitCode::from(2)
        }
        Some("--version") | Some("-V") => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            println!("Lumen — self-hosted LLM inference compiler\n");
            println!("USAGE:");
            println!("  lumen run <model.gguf> <prompt>");
            println!("  lumen compile <input.lum> -o <output>");
            println!("  lumen bench <model.gguf>");
            println!("  lumen --version");
            ExitCode::SUCCESS
        }
    }
}
