//! `lumen` — command-line entry point.
//!
//! Subcommands:
//!   lumen parse <input.lum>        # lex + parse, dump AST
//!   lumen check <input.lum>        # lex + parse + type-check
//!   lumen run <model.gguf> <text>  # (Phase 6)
//!   lumen compile <input.lum>      # (Phase 2)
//!   lumen bench <model.gguf>       # (Phase 7)

use std::env;
use std::fs;
use std::process::ExitCode;

use lumen_dsl::{render_all, Parser, TypeChecker};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("parse") => cmd_parse(&args),
        Some("check") => cmd_check(&args),
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
            print_usage();
            ExitCode::SUCCESS
        }
    }
}

fn print_usage() {
    println!("Lumen — self-hosted LLM inference compiler\n");
    println!("USAGE:");
    println!("  lumen parse <input.lum>          Lex + parse + dump AST");
    println!("  lumen check <input.lum>          Lex + parse + type-check");
    println!("  lumen run <model.gguf> <prompt>  (Phase 6)");
    println!("  lumen compile <input.lum>        (Phase 2)");
    println!("  lumen bench <model.gguf>         (Phase 7)");
    println!("  lumen --version");
}

fn cmd_parse(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2) else {
        eprintln!("error: missing input file");
        eprintln!("usage: lumen parse <input.lum>");
        return ExitCode::from(2);
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path, e);
            return ExitCode::from(1);
        }
    };
    match Parser::parse(&source) {
        Ok(module) => {
            println!("{:#?}", module);
            ExitCode::SUCCESS
        }
        Err(errs) => {
            eprint!("{}", render_all(&errs, &source, path));
            ExitCode::from(1)
        }
    }
}

fn cmd_check(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2) else {
        eprintln!("error: missing input file");
        eprintln!("usage: lumen check <input.lum>");
        return ExitCode::from(2);
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path, e);
            return ExitCode::from(1);
        }
    };
    let module = match Parser::parse(&source) {
        Ok(m) => m,
        Err(errs) => {
            eprint!("{}", render_all(&errs, &source, path));
            return ExitCode::from(1);
        }
    };
    if let Err(errs) = TypeChecker::check(&module) {
        eprint!("{}", render_all(&errs, &source, path));
        return ExitCode::from(1);
    }
    println!("ok: {} type-checked", path);
    ExitCode::SUCCESS
}
