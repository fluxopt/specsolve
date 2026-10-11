// A specsolve wheel, installed into Pyodide the way the browser installs it, then probe.py run against it.
//
//   pixi run test-pyodide
//
// micropip resolves the wheel's requirements against this Pyodide's own packages,
// so a requirement the browser cannot meet fails the install by name.
import { readFileSync } from 'node:fs'
import { basename } from 'node:path'

import { loadPyodide, version } from 'pyodide'

const [wheel] = process.argv.slice(2)
const pyodide = await loadPyodide({ packageBaseUrl: `https://cdn.jsdelivr.net/pyodide/v${version}/full/` })
await pyodide.loadPackage('micropip')
const at = `/tmp/${basename(wheel)}`
pyodide.FS.writeFile(at, readFileSync(wheel))
await pyodide.pyimport('micropip').install(`emfs:${at}`)
pyodide.FS.writeFile('/home/pyodide/probe.py', readFileSync(new URL('probe.py', import.meta.url)))
console.log(`Pyodide ${version}: ${pyodide.runPython('import probe; probe.main()')}`)
