// A specsolve wheel, installed into Pyodide the way the browser installs it, then probe.py run against it.
//
//   pixi run test-pyodide
//
// micropip resolves the wheel's requirements against this Pyodide's own packages,
// so a requirement the browser cannot meet fails the install by name. A wheel that
// direct.py built beside it, under direct/, is installed first: a dependency
// pyproject.toml references by git commit, which micropip cannot fetch.
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'

import { loadPyodide, version } from 'pyodide'

const [wheel] = process.argv.slice(2)
const pyodide = await loadPyodide({ packageBaseUrl: `https://cdn.jsdelivr.net/pyodide/v${version}/full/` })
await pyodide.loadPackage('micropip')
const direct = join(dirname(wheel), 'direct')
const first = existsSync(direct) ? readdirSync(direct).filter((f) => f.endsWith('.whl')).map((f) => join(direct, f)) : []
for (const path of [...first, wheel]) {
  const at = `/tmp/${basename(path)}`
  pyodide.FS.writeFile(at, readFileSync(path))
  await pyodide.pyimport('micropip').install(`emfs:${at}`)
}
pyodide.FS.writeFile('/home/pyodide/probe.py', readFileSync(new URL('probe.py', import.meta.url)))
console.log(`Pyodide ${version}: ${pyodide.runPython('import probe; probe.without_pyarrow()')}`)
await pyodide.loadPackage(['pyarrow'])
console.log(`with pyarrow: ${pyodide.runPython('probe.with_pyarrow()')}`)
