import { runAi } from './modules/AI';
import { uploadButton } from './modules/Dom';
import './style.css'

uploadButton.addEventListener("click", runAi)