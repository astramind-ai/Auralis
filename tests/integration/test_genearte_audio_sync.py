#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

import asyncio
import os.path

import pytest
import torch

from auralis.common.definitions.dto.requests import TTSRequest
from auralis.models.xttsv2.XTTSv2 import XTTSv2Engine
from auralis.core.tts import TTS

text = """La Storia di Villa Margherita

L'antica Villa Margherita si ergeva maestosa sulla collina che dominava il piccolo paese di San Lorenzo, circondata da cipressi centenari e giardini rigogliosi. Era stata costruita nel 1876 dal Conte Alessandro Visconti per la sua giovane sposa, Margherita, da cui prese il nome.

La villa aveva attraversato diverse epoche, resistendo a due guerre mondiali e numerose trasformazioni sociali. Durante la Seconda Guerra Mondiale, era diventata un rifugio per molte famiglie che fuggivano dai bombardamenti sulla città. Le sue spesse mura di pietra avevano protetto centinaia di vite.

"Non posso credere che vogliano demolirla!" esclamò Maria, l'ultima discendente dei Visconti, durante un'accesa discussione con il consiglio comunale. "Questa villa non è solo un edificio, è la memoria vivente della nostra comunità!"

Il sindaco, seduto dietro la sua scrivania di quercia, sospirò profondamente. "Capisco il suo punto di vista, signorina Visconti, ma i costi di restauro sono proibitivi. La struttura è pericolante, e non possiamo permetterci di rischiare incidenti."

La notizia della possibile demolizione si diffuse rapidamente nel paese, suscitando reazioni contrastanti. Alcuni abitanti sostenevano che fosse giunto il momento di fare spazio a strutture più moderne, mentre altri consideravano la villa un patrimonio culturale insostituibile.

Anna, la bibliotecaria comunale che aveva trascorso l'infanzia giocando nei giardini della villa, decise di agire. Organizzò una raccolta fondi e coinvolse esperti di restauro da tutta Italia. "Se lavoriamo insieme," disse durante un'assemblea pubblica, "possiamo salvare questo pezzo di storia!"

Le settimane seguenti furono caratterizzate da un fermento straordinario. Artigiani locali si offrirono volontari per i lavori di restauro meno specializzati. Gli studenti dell'istituto d'arte realizzarono un documentario sulla storia della villa. Persino alcune aziende della regione decisero di sponsorizzare il progetto.

Il professor Martelli, storico dell'architettura dell'università vicina, scoprì nei sotterranei della villa una collezione di lettere e documenti che rivelavano dettagli inediti sulla sua storia. "Queste carte dimostrano che Villa Margherita fu un importante centro culturale all'inizio del '900," spiegò entusiasta. "Ospitò artisti, scrittori e musicisti da tutta Europa!"

Dopo mesi di lavoro instancabile, la villa cominciò a riprendere il suo antico splendore. Le pareti furono consolidate, il tetto riparato, e i giardini riportati all'originale bellezza. Durante i lavori, gli operai scoprirono affreschi nascosti sotto strati di intonaco e una piccola cappella che era stata murata.

Il giorno della riapertura, l'intera comunità si riunì per celebrare. Maria Visconti, con le lacrime agli occhi, tagliò il nastro rosso all'ingresso. "Oggi non festeggiamo solo il restauro di un edificio," disse commossa, "ma la rinascita di un luogo che continuerà a raccontare storie per le generazioni future."

La villa divenne un museo e centro culturale, ospitando mostre, concerti e laboratori per bambini. Le sue sale, che un tempo rischiavano di essere ridotte in macerie, tornarono a riempirsi di vita e di voci. I giardini diventarono un parco pubblico, dove gli abitanti del paese potevano passeggiare e rilassarsi.

"È incredibile come un edificio possa unire così tante persone," commentò Anna, guardando un gruppo di studenti che disegnavano nel giardino. "Villa Margherita non è più solo un monumento del passato, ma un simbolo di cosa possiamo realizzare quando lavoriamo insieme per un obiettivo comune."

E così, grazie all'impegno di un'intera comunità, Villa Margherita continuò a dominare la collina di San Lorenzo, non più come un peso da sostenere, ma come un faro di cultura e memoria collettiva, pronta ad accogliere nuove storie e nuovi sogni.

?!... "Davvero sorprendente!" esclamò un visitatore. "Non avrei mai immaginato che un edificio potesse racchiudere tante storie." Le sue parole echeggiarono nelle sale della villa, mescolandosi con i sussurri dei visitatori e il fruscio delle foglie dei cipressi centenari che, come sempre, montavano la guardia alla memoria di quel luogo straordinario."""

speaker_file = "./female.wav"
# Inizializza il TTS
tts = TTS()
tts.tts_engine = XTTSv2Engine.from_pretrained("AstraMindAI/xtts2", torch_dtype=torch.float32)

def test_tts_sync_generator():
    # Create a TTS request
    request_for_generator = TTSRequest(
        text=text,
        language="it",
        speaker_files=["/home/marco/PycharmProjects/betterVoiceCraft/female.wav"],
        stream=True
    )
    generator = tts.generate_speech(request_for_generator)

    output_list = []
    # Consume the results
    for result in generator:
        if isinstance(result, Exception):
            raise result
        else:
            output_list.append(result)

    assert len(output_list) > 0


def test_tts_generation():
    # Create a TTS request
    request_for_generator = TTSRequest(
        text=text,
        language="it",
        speaker_files=[os.path.join(os.path.dirname(__file__), '..', 'resources', 'audio_samples', 'female.wav')],
        stream=False
    )
    audio = tts.generate_speech(request_for_generator)

    assert len(audio.array) > 0
