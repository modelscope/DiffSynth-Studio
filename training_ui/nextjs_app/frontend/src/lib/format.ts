export function formatDateTime(value?: string | null): string {
  if (!value) return "-";
  return String(value).replace("T", " ").split(".", 1)[0];
}

function quoteShellArg(value: string): string {
  return /[\s"'*?$`\\<>|&;()]/.test(value)
    ? `'${value.replace(/'/g, `'\\''`)}'`
    : value;
}

function splitShellStages(command: string): string[] {
  const stages: string[] = [];
  let current = "";
  let quote: "'" | '"' | null = null;
  let escaped = false;

  for (let index = 0; index < command.length; index += 1) {
    const character = command[index];
    if (escaped) {
      current += character;
      escaped = false;
      continue;
    }
    if (character === "\\" && quote !== "'") {
      current += character;
      escaped = true;
      continue;
    }
    if (character === "'" || character === '"') {
      if (quote === character) quote = null;
      else if (quote === null) quote = character;
      current += character;
      continue;
    }
    if (quote === null && command.startsWith(" && ", index)) {
      stages.push(current.trim());
      current = "";
      index += 3;
      continue;
    }
    current += character;
  }

  if (current.trim()) stages.push(current.trim());
  return stages;
}

function formatJoinedShellCommand(command: string): string {
  const continuation = " \\" + "\n  ";

  return splitShellStages(command).map((stage) => {
    let formatted = "";
    let quote: "'" | '"' | null = null;
    let escaped = false;

    for (let index = 0; index < stage.length; index += 1) {
      const character = stage[index];
      if (escaped) {
        formatted += character;
        escaped = false;
        continue;
      }
      if (character === "\\" && quote !== "'") {
        formatted += character;
        escaped = true;
        continue;
      }
      if (character === "'" || character === '"') {
        if (quote === character) quote = null;
        else if (quote === null) quote = character;
        formatted += character;
        continue;
      }
      if (quote === null && stage.startsWith(" --", index)) {
        formatted += `${continuation}--`;
        index += 2;
        continue;
      }
      formatted += character;
    }
    return formatted;
  }).join(" &&\n\n");
}

export function formatShellCommand(argv: string[]): string {
  if (argv.length === 0) return "";

  if (argv.length === 3 && argv[0] === "bash" && argv[1] === "-lc") {
    return formatJoinedShellCommand(argv[2]);
  }

  const continuation = " \\" + "\n  ";
  const firstOption = argv.findIndex((value) => value.startsWith("--"));
  if (firstOption < 0) {
    if (argv.length === 1) return quoteShellArg(argv[0]);
    return argv.slice(0, -1).map(quoteShellArg).join(" ")
      + continuation
      + quoteShellArg(argv[argv.length - 1]);
  }

  const lines = [argv.slice(0, firstOption).map(quoteShellArg).join(" ")];
  let index = firstOption;
  while (index < argv.length) {
    const option = argv[index];
    const value = argv[index + 1];
    if (option.startsWith("--") && value !== undefined && !value.startsWith("--")) {
      lines.push(`${quoteShellArg(option)} ${quoteShellArg(value)}`);
      index += 2;
    } else {
      lines.push(quoteShellArg(option));
      index += 1;
    }
  }
  return lines.join(continuation);
}
