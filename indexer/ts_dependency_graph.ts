import { Project } from "ts-morph";
import * as fs from "fs";
import * as path from "path";
import yaml from "js-yaml";

interface Config {
  project_path: string;
  ignore?: string[];
}

function loadConfig(): Config {
  const raw = fs.readFileSync(path.join(__dirname, "config.yaml"), "utf8");
  return yaml.load(raw) as Config;
}

export interface DependencyGraph {
  [file: string]: string[];
}

export function buildDependencyGraph(): DependencyGraph {
  const config = loadConfig();
  const basePath = path.resolve(config.project_path);
  const ignore = config.ignore ?? [];

  const project = new Project({
    tsConfigFilePath: fs.existsSync(path.join(basePath, "tsconfig.json"))
      ? path.join(basePath, "tsconfig.json")
      : undefined,
    skipFileDependencyResolution: false,
  });

  const patterns = [`${basePath}/**/*.ts`, `${basePath}/**/*.tsx`];
  for (const pattern of ignore) {
    patterns.push(`!${path.join(basePath, pattern)}/**`);
  }

  project.addSourceFilesAtPaths(patterns);

  const graph: DependencyGraph = {};
  for (const sourceFile of project.getSourceFiles()) {
    const filePath = path.relative(basePath, sourceFile.getFilePath());
    const imports = sourceFile
      .getImportDeclarations()
      .map((imp) => imp.getModuleSpecifierSourceFile())
      .filter((sf): sf is ReturnType<typeof sourceFile.getImportDeclarations>[0]["getModuleSpecifierSourceFile"] => !!sf)
      .map((sf) => path.relative(basePath, sf.getFilePath()));

    graph[filePath] = Array.from(new Set(imports));
  }

  return graph;
}

if (require.main === module) {
  const graph = buildDependencyGraph();
  const outPath = path.join(__dirname, "dependency_graph.json");
  fs.writeFileSync(outPath, JSON.stringify(graph, null, 2), "utf8");
  // eslint-disable-next-line no-console
  console.log(`Dependency graph written to ${outPath}`);
}
