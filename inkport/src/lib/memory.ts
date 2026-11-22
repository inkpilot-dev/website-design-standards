/**
 * Will training fit in this computer's memory?
 *
 * On a graphics card or Apple Silicon the model is held in a compact form. On the processor it is
 * held in full precision: 4 bytes for each parameter, plus working memory. The included 1.7B
 * model therefore needs about 9 GB, more than many laptops have free. Training still runs with
 * less (the system pages memory to disk), but slowly, and the whole computer becomes sluggish.
 * Saying so before the user presses Start is better than letting them find out.
 */
export interface MemoryNote {
  /** "short": less memory than the run needs; "tight": enough, with little to spare. */
  level: "short" | "tight";
  needGb: number;
  ramGb: number;
  text: string;
}

export function trainingMemoryNote(ramGb: number | null | undefined, paramsB: number | null | undefined, accelerator: string | null | undefined): MemoryNote | null {
  if (!ramGb || !paramsB || !accelerator || !/processor|cpu/i.test(accelerator)) return null;
  const needGb = Math.ceil(paramsB * 4 + 1.5);
  const have = Math.round(ramGb);
  if (ramGb >= needGb + 3) return null;
  if (ramGb < needGb) {
    return { level: "short", needGb, ramGb: have, text: `This computer has ${have} GB of memory, and training this model on the processor needs about ${needGb} GB. Training will still run, but slowly, and the computer may respond sluggishly until it finishes. Close other programs before you start.` };
  }
  return { level: "tight", needGb, ramGb: have, text: `Training this model on the processor needs about ${needGb} GB of this computer's ${have} GB of memory. Close other programs before you start, so it does not slow down.` };
}
