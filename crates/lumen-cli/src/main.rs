//! `lumen` — command-line entry point.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use lumen_codegen::emit_c;
use lumen_dsl::{render_all, Parser, TypeChecker};
use lumen_ir::{lower, print_module, verify_module};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("parse") => cmd_parse(&args),
        Some("check") => cmd_check(&args),
        Some("ir") => cmd_ir(&args),
        Some("compile-c") => cmd_compile_c(&args),
        Some("run") => {
            eprintln!("lumen run: not implemented yet (Phase 6)");
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
    println!("  lumen parse     <input.lum>             Lex + parse + dump AST");
    println!("  lumen check     <input.lum>             Lex + parse + type-check");
    println!("  lumen ir        <input.lum>             Dump IR (after lowering + verify)");
    println!("  lumen compile-c <input.lum> -o <out.c>  Emit C source from IR");
    println!("  lumen run       <model.gguf> <prompt>   (Phase 6)");
    println!("  lumen bench     <model.gguf>            (Phase 7)");
    println!("  lumen --version");
}

fn read_source(path: &str) -> Result<String, ExitCode> {
    fs::read_to_string(path).map_err(|e| {
        eprintln!("error: cannot read {}: {}", path, e);
        ExitCode::from(1)
    })
}

fn cmd_parse(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2) else {
        eprintln!("usage: lumen parse <input.lum>");
        return ExitCode::from(2);
    };
    let source = match read_source(path) {
        Ok(s) => s,
        Err(c) => return c,
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
        eprintln!("usage: lumen check <input.lum>");
        return ExitCode::from(2);
    };
    let source = match read_source(path) {
        Ok(s) => s,
        Err(c) => return c,
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

fn cmd_ir(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2) else {
        eprintln!("usage: lumen ir <input.lum>");
        return ExitCode::from(2);
    };
    let source = match read_source(path) {
        Ok(s) => s,
        Err(c) => return c,
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
    let ir = match lower(&module) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("error: lowering failed: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(errs) = verify_module(&ir) {
        for e in errs {
            eprintln!("verify error: {}", e);
        }
        return ExitCode::from(1);
    }
    print!("{}", print_module(&ir));
    ExitCode::SUCCESS
}

fn cmd_compile_c(args: &[String]) -> ExitCode {
    let Some(path) = args.get(2) else {
        eprintln!("usage: lumen compile-c <input.lum> [-o <out.c>]");
        return ExitCode::from(2);
    };
    let out_path: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]));

    let source = match read_source(path) {
        Ok(s) => s,
        Err(c) => return c,
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
    let ir = match lower(&module) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("error: lowering failed: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(errs) = verify_module(&ir) {
        for e in errs {
            eprintln!("verify error: {}", e);
        }
        return ExitCode::from(1);
    }
    let c_src = match emit_c(&ir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: C backend: {}", e);
            return ExitCode::from(1);
        }
    };

    match out_path {
        Some(p) => {
            if let Err(e) = fs::write(&p, &c_src) {
                eprintln!("error: cannot write {}: {}", p.display(), e);
                return ExitCode::from(1);
            }
            println!("wrote {}", p.display());
        }
        None => {
            print!("{}", c_src);
        }
    }
    ExitCode::SUCCESS
}
